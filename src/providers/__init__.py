"""
Model Provider Abstraction Layer

Provides a unified interface for multiple LLM providers:
- OpenRouter (gateway to multiple models)
- Google Gemini
- Anthropic Claude

Usage:
    from providers import get_provider, ModelTier

    # Get tier-based provider
    provider = get_provider(ModelTier.TIER1)  # Fast/cheap model
    provider = get_provider(ModelTier.TIER2)  # Quality model

    # Or specific provider
    from providers.openrouter import OpenRouterProvider
    provider = OpenRouterProvider(model="qwen/qwen-2.5-72b-instruct")
"""

from __future__ import annotations

from providers.base import (
    ModelProvider,
    CompletionResult,
    CompletionRequest,
    TokenUsage,
    ModelInfo,
    ProviderError,
    RateLimitError,
    AuthenticationError,
)
from providers.config import (
    ModelTier,
    ProviderConfig,
    get_config,
    get_provider,
    get_model_info,
)

__all__ = [
    # Base types
    "ModelProvider",
    "CompletionResult",
    "CompletionRequest",
    "TokenUsage",
    "ModelInfo",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    # Config
    "ModelTier",
    "ProviderConfig",
    "get_config",
    "get_provider",
    "get_model_info",
]
