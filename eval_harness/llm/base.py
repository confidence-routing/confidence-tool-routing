"""Provider-independent contract for text generation."""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    """A minimal text-generation client suitable for dependency injection."""

    def generate(self, prompt: str) -> str:
        """Generate an assistant response for ``prompt``."""
