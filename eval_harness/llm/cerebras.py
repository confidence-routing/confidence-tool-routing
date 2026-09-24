"""Cerebras implementation of the eval harness LLM contract."""

from __future__ import annotations

import os
from typing import Any

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:  # Keep importing the harness possible without optional setup.
    Cerebras = None  # type: ignore[assignment,misc]


class CerebrasClient:
    """Generate text through Cerebras chat completions."""

    def __init__(self, model: str):
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not configured. Set it before creating a CerebrasClient."
            )
        if Cerebras is None:
            raise ImportError(
                "cerebras_cloud_sdk is required to use CerebrasClient. "
                "Install dependencies with 'pip install -r requirements.txt'."
            )

        self.model = model
        self._client: Any = Cerebras(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Return the assistant text from a Cerebras chat completion."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
