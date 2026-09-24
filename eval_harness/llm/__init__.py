"""Small LLM interfaces and provider implementations for the eval harness."""

from .base import LLM
from .cerebras import CerebrasClient

__all__ = ["LLM", "CerebrasClient"]
