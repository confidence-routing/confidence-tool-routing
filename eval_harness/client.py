"""
client.py
=========

The one place that makes a network call.

Everything else in this package is pure -- estimators score responses
that already exist, metrics aggregate records that already exist. This
module is the boundary, and it is deliberately thin: it sends a chat
completion and hands back whatever came off the wire, unwrapped.

Provider-agnostic by construction
---------------------------------
OpenAI, Cerebras and OpenRouter all speak the same chat-completions
shape, so there is one code path and a base URL, not three classes.
Cerebras and OpenRouter matter here because their free tiers return
logprobs, which is the whole ballgame: without logprobs the token-entropy
estimator has no signal, and it is the only one of the four that costs
nothing.

Provider differences that actually bite:

  n            Cerebras accepts only n=1, so self-consistency has to make
               k separate calls rather than one k-choice request.
               sample() does that, and extract_sampled_answers() already
               reads both shapes, so nothing downstream changes.
  logprobs     Documented by all three, but Cerebras warns that parameter
               support varies by model. probe() checks a given model
               rather than trusting the docs -- see __main__ below.
  rate limits  Free tiers throttle hard. complete() retries on 429 and
               5xx with exponential backoff, because a sweep that dies
               two thirds through is a sweep you pay for twice.

Responses are returned raw. The estimators read both dicts and objects
(see _get in confidence.py), so wrapping them in a dataclass here would
buy nothing and cost a translation layer that could drop a field.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------
# The MissingAPIKey message tells you to put the key in a .env, so
# something has to read one. Twelve lines of stdlib rather than a
# python-dotenv dependency: this is the only place in the harness that
# needs it, and requirements.txt is one package for a reason.
#
# Searched upwards from the package, because the repo is often a
# subdirectory of the checkout and the .env lands next to it rather than
# inside it. Environment variables always win over the file -- an
# explicit export should not be silently overridden by a stale .env.
# ---------------------------------------------------------------------------

_DOTENV_SEARCH_DEPTH = 3
_dotenv_loaded = False


def load_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """Load the nearest .env into os.environ. Returns the file used, if any."""
    global _dotenv_loaded

    here = (start or Path(__file__).resolve().parent)
    for directory in [here, *here.parents][:_DOTENV_SEARCH_DEPTH + 1]:
        candidate = directory / ".env"
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Never clobber a real environment variable.
            if key and key not in os.environ:
                os.environ[key] = value
        _dotenv_loaded = True
        return candidate
    return None


# Base URLs for the OpenAI-compatible endpoints. The env var names are
# each provider's own convention, so an existing key works unchanged.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai":     {"base_url": "",  # the SDK default
                   "env": "OPENAI_API_KEY"},
    "cerebras":   {"base_url": "https://api.cerebras.ai/v1",
                   "env": "CEREBRAS_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "env": "OPENROUTER_API_KEY"},
    # Local inference, OpenAI-compatible. Ollama DOES return logprobs and
    # top_logprobs despite its compatibility doc not listing them -- checked
    # against 0.34.4, and the response shape is exactly what
    # confidence.py's extract_token_logprobs() expects.
    #
    # No key is needed, so the env var is a placeholder the SDK accepts;
    # _needs_key marks the provider as keyless rather than special-casing
    # the constructor.
    "ollama":     {"base_url": "http://localhost:11434/v1",
                   "env": "OLLAMA_API_KEY", "needs_key": "no"},
}

DEFAULT_PROVIDER = "cerebras"

# Retry policy. Free tiers 429 readily; these are the codes worth
# retrying, as opposed to a 400 that will fail identically every time.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0


class MissingAPIKey(RuntimeError):
    pass


def _status_of(exc: Exception) -> Optional[int]:
    """HTTP status off an SDK exception, whatever shape it arrived in."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


@dataclass
class Client:
    """A configured chat-completions caller for one provider."""

    provider: str = DEFAULT_PROVIDER
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_attempts: int = _MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {self.provider!r}. "
                f"Known: {', '.join(sorted(PROVIDERS))}"
            )
        spec = PROVIDERS[self.provider]

        key = self.api_key or os.environ.get(spec["env"])
        if not key and not _dotenv_loaded:
            load_dotenv()
            key = os.environ.get(spec["env"])
        if not key and spec.get("needs_key") == "no":
            # A local server authenticates nothing; the SDK still wants a
            # non-empty string.
            key = "local"
        if not key:
            raise MissingAPIKey(
                f"No API key for provider {self.provider!r}. "
                f"Set {spec['env']} in the environment (or a .env, which "
                f"is gitignored) and try again."
            )
        self.api_key = key

        if self.base_url is None:
            self.base_url = spec["base_url"] or None

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise ImportError(
                "The 'openai' package is required to make calls. "
                "pip install -r requirements.txt"
            ) from exc

        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._sdk = OpenAI(**kwargs)

    # -- the call ----------------------------------------------------------

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        *,
        logprobs: bool = True,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        **extra: Any,
    ) -> Any:
        """
        One chat completion, returned exactly as the provider sent it.

        logprobs defaults to True because they are free -- metadata on a
        call you were making anyway -- and they are what the entropy
        estimator reads. Turning them off silently removes an estimator
        from the experiment, so the default is the one that keeps it.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        if seed is not None:
            payload["seed"] = seed
        payload.update(extra)

        last: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                return self._sdk.chat.completions.create(**payload)
            except Exception as exc:  # SDK exception types vary by version
                status = _status_of(exc)
                if status is not None and status not in _RETRY_STATUS:
                    raise
                last = exc
                if attempt == self.max_attempts - 1:
                    break
                # Exponential backoff with jitter: a sweep hitting a
                # shared free-tier limit should not resynchronise its
                # retries into another burst.
                delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                time.sleep(delay * (0.5 + random.random()))

        raise RuntimeError(
            f"{self.provider} call failed after {self.max_attempts} attempts: {last}"
        ) from last

    def sample(
        self,
        messages: List[Dict[str, str]],
        model: str,
        *,
        k: int = 5,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> List[Any]:
        """
        k independent samples, for the self-consistency estimator.

        Always k separate calls, never one n=k request. Cerebras caps n
        at 1, and the cost accounting is cleaner this way: every sample
        reports its own usage, so the confidence-stage token count is
        summed from what was actually billed rather than divided out of
        an aggregate.

        Temperature defaults non-zero on purpose. At temperature 0 every
        sample is the same sample and agreement is 1.0 by construction --
        a perfectly confident, perfectly meaningless signal.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if temperature <= 0:
            raise ValueError(
                "self-consistency needs temperature > 0; at 0 every sample "
                "is identical and agreement is 1.0 regardless of the model"
            )
        return [
            self.complete(messages, model, temperature=temperature,
                          seed=None, **kwargs)
            for _ in range(k)
        ]

    # -- capability probe --------------------------------------------------

    def probe(self, model: str) -> Dict[str, Any]:
        """
        Does THIS model on THIS provider actually return logprobs?

        Cerebras documents logprobs but warns that parameter support
        varies by model, and OpenRouter's support depends on whichever
        provider is serving the request. Both mean the docs cannot
        settle it -- one cheap call can. Run this before a sweep rather
        than discovering half way through that every confidence_score
        came back None.
        """
        from .confidence import extract_token_logprobs, _get

        result: Dict[str, Any] = {
            "provider": self.provider, "model": model,
            "ok": False, "logprobs": False, "top_logprobs": False,
            "error": None,
        }
        try:
            response = self.complete(
                [{"role": "user", "content": "Reply with the single word: yes"}],
                model, max_tokens=5,
            )
        except Exception as exc:
            result["error"] = str(exc)
            return result

        result["ok"] = True
        entries = extract_token_logprobs(response)
        result["logprobs"] = bool(entries)
        if entries:
            top = _get(entries[0], "top_logprobs")
            result["top_logprobs"] = bool(top)
        usage = _get(response, "usage")
        result["usage"] = {
            "prompt_tokens": _get(usage, "prompt_tokens"),
            "completion_tokens": _get(usage, "completion_tokens"),
        }
        return result


def usage_tokens(response: Any) -> Dict[str, int]:
    """(prompt, completion) token counts off a response, zeros if absent.

    Every stage's cost is summed from what the provider actually billed
    rather than re-estimated with a tokenizer, so a provider that counts
    differently than tiktoken does not silently skew CPST.
    """
    from .confidence import _get

    usage = _get(response, "usage")
    return {
        "prompt_tokens": int(_get(usage, "prompt_tokens") or 0),
        "completion_tokens": int(_get(usage, "completion_tokens") or 0),
    }


if __name__ == "__main__":  # pragma: no cover
    # Capability check: python -m eval_harness.client [provider] [model]
    import json
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROVIDER
    model = sys.argv[2] if len(sys.argv) > 2 else "llama-3.3-70b"
    print(json.dumps(Client(provider=provider).probe(model), indent=2))
