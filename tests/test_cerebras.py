"""Offline unit tests for the Cerebras LLM adapter."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness.llm.cerebras import CerebrasClient


def _fake_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_missing_api_key_raises_clear_error():
    with patch.dict(os.environ, {}, clear=True):
        try:
            CerebrasClient(model="configured-model")
        except RuntimeError as exc:
            assert "CEREBRAS_API_KEY" in str(exc)
            print("test_missing_api_key_raises_clear_error: PASS")
            return
    raise AssertionError("expected a missing API key to raise RuntimeError")


def test_initialization_passes_environment_api_key_to_sdk():
    fake_sdk = patch("eval_harness.llm.cerebras.Cerebras")
    with patch.dict(os.environ, {"CEREBRAS_API_KEY": "test-key"}, clear=True), fake_sdk as sdk:
        client = CerebrasClient(model="configured-model")

    assert client.model == "configured-model"
    sdk.assert_called_once_with(api_key="test-key")
    print("test_initialization_passes_environment_api_key_to_sdk: PASS")


def test_generate_sends_prompt_and_model_and_returns_assistant_content():
    fake_sdk = patch("eval_harness.llm.cerebras.Cerebras")
    with patch.dict(os.environ, {"CEREBRAS_API_KEY": "test-key"}, clear=True), fake_sdk as sdk:
        sdk.return_value.chat.completions.create.return_value = _fake_response("assistant answer")
        client = CerebrasClient(model="configured-model")
        answer = client.generate("What is the answer?")

    assert answer == "assistant answer"
    sdk.return_value.chat.completions.create.assert_called_once_with(
        model="configured-model",
        messages=[{"role": "user", "content": "What is the answer?"}],
    )
    print("test_generate_sends_prompt_and_model_and_returns_assistant_content: PASS")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
