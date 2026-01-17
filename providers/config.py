#!/usr/bin/env python3
"""
Centralized configuration for model providers.

Manages model selection, pricing tables, and default parameters.
Supports two-tier model strategy:
- Tier 1: Fast/cheap models for parallel chunk extraction
- Tier 2: Quality models for final synthesis
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from providers.base import ModelInfo, ModelProvider

if TYPE_CHECKING:
    from providers.anthropic import AnthropicProvider
    from providers.gemini import GeminiProvider
    from providers.openrouter import OpenRouterProvider


# =============================================================================
# Model Tiers
# =============================================================================


class ModelTier(str, Enum):
    """Model tier for two-tier extraction strategy."""

    TIER1 = "tier1"  # Fast/cheap for chunk extraction
    TIER2 = "tier2"  # Quality for synthesis


class ProviderType(str, Enum):
    """Available provider types."""

    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


# =============================================================================
# Default Model Configurations
# =============================================================================

# Default models per tier
DEFAULT_TIER1_MODEL = "google/gemini-2.0-flash-001"  # Fast, cheap via OpenRouter
DEFAULT_TIER2_MODEL = "anthropic/claude-sonnet-4"  # Quality via OpenRouter

# Alternative tier configurations
TIER_ALTERNATIVES = {
    ModelTier.TIER1: [
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
        "openai/gpt-4o-mini",
        "qwen/qwen-2.5-72b-instruct",
    ],
    ModelTier.TIER2: [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o",
    ],
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ProviderConfig:
    """Configuration for a specific provider instance."""

    provider_type: ProviderType
    model_id: str
    api_key: str

    # Request parameters
    temperature: float = 0.3
    max_tokens: int | None = None
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 300.0

    # OpenRouter specific
    api_url: str | None = None

    def to_dict(self) -> dict[str, str | float | int | None]:
        """Serialize to dict (excluding api_key)."""
        return {
            "provider_type": self.provider_type.value,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
        }


@dataclass
class PipelineConfig:
    """Configuration for the full extraction pipeline."""

    tier1_config: ProviderConfig
    tier2_config: ProviderConfig

    # Chunking parameters
    max_tokens_per_chunk: int = 800
    chunk_overlap_ratio: float = 0.15
    temporal_gap_minutes: int = 20

    # Extraction parameters
    max_parallel_extractions: int = 5

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict."""
        return {
            "tier1": self.tier1_config.to_dict(),
            "tier2": self.tier2_config.to_dict(),
            "max_tokens_per_chunk": self.max_tokens_per_chunk,
            "chunk_overlap_ratio": self.chunk_overlap_ratio,
            "temporal_gap_minutes": self.temporal_gap_minutes,
            "max_parallel_extractions": self.max_parallel_extractions,
        }


# =============================================================================
# Configuration Loading
# =============================================================================


def _detect_provider_type(model_id: str) -> ProviderType:
    """Detect provider type from model ID."""
    # OpenRouter models have provider prefix
    if "/" in model_id:
        return ProviderType.OPENROUTER

    # Gemini models start with "gemini"
    if model_id.startswith("gemini"):
        return ProviderType.GEMINI

    # Claude models start with "claude"
    if model_id.startswith("claude"):
        return ProviderType.ANTHROPIC

    # Default to OpenRouter
    return ProviderType.OPENROUTER


def _get_api_key_for_provider(provider_type: ProviderType) -> str:
    """Get API key for provider from environment."""
    load_dotenv()

    key_map = {
        ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
        ProviderType.GEMINI: "GOOGLE_API_KEY",
        ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    }

    env_var = key_map.get(provider_type, "OPENROUTER_API_KEY")
    api_key = os.getenv(env_var)

    if not api_key:
        raise ValueError(f"API key not found: {env_var}")

    return api_key


@lru_cache(maxsize=1)
def get_config() -> PipelineConfig:
    """
    Load pipeline configuration from environment.

    Environment variables:
        TIER1_MODEL: Model ID for tier 1 (default: google/gemini-1.5-flash)
        TIER2_MODEL: Model ID for tier 2 (default: anthropic/claude-3.5-sonnet)
        OPENROUTER_API_KEY: OpenRouter API key
        GOOGLE_API_KEY: Google AI API key (for direct Gemini)
        ANTHROPIC_API_KEY: Anthropic API key (for direct Claude)
        OPENROUTER_TEMPERATURE: Default temperature (default: 0.3)
        OPENROUTER_MAX_TOKENS: Max tokens (default: none)
        OPENROUTER_MAX_RETRIES: Max retries (default: 3)
        OPENROUTER_RETRY_DELAY: Retry delay (default: 5)
        OPENROUTER_API_URL: Custom API URL

    Returns:
        PipelineConfig with loaded settings.
    """
    load_dotenv()

    # Get model IDs
    tier1_model = os.getenv("TIER1_MODEL", DEFAULT_TIER1_MODEL)
    tier2_model = os.getenv("TIER2_MODEL", DEFAULT_TIER2_MODEL)

    # Detect provider types
    tier1_provider = _detect_provider_type(tier1_model)
    tier2_provider = _detect_provider_type(tier2_model)

    # Get common parameters
    temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
    max_tokens_str = os.getenv("OPENROUTER_MAX_TOKENS", "none")
    max_tokens = None if max_tokens_str.lower() == "none" else int(max_tokens_str)
    max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("OPENROUTER_RETRY_DELAY", "5"))
    api_url = os.getenv("OPENROUTER_API_URL")

    # Create tier configs
    tier1_config = ProviderConfig(
        provider_type=tier1_provider,
        model_id=tier1_model,
        api_key=_get_api_key_for_provider(tier1_provider),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
        api_url=api_url if tier1_provider == ProviderType.OPENROUTER else None,
    )

    tier2_config = ProviderConfig(
        provider_type=tier2_provider,
        model_id=tier2_model,
        api_key=_get_api_key_for_provider(tier2_provider),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
        api_url=api_url if tier2_provider == ProviderType.OPENROUTER else None,
    )

    return PipelineConfig(
        tier1_config=tier1_config,
        tier2_config=tier2_config,
    )


# =============================================================================
# Provider Factory
# =============================================================================


def create_provider(config: ProviderConfig) -> ModelProvider:
    """
    Create a provider instance from configuration.

    Args:
        config: Provider configuration.

    Returns:
        Configured ModelProvider instance.
    """
    if config.provider_type == ProviderType.OPENROUTER:
        from providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(
            model_id=config.model_id,
            api_key=config.api_key,
            api_url=config.api_url,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            timeout=config.timeout,
        )

    elif config.provider_type == ProviderType.GEMINI:
        from providers.gemini import GeminiProvider
        return GeminiProvider(
            model_id=config.model_id,
            api_key=config.api_key,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            timeout=config.timeout,
        )

    elif config.provider_type == ProviderType.ANTHROPIC:
        from providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            model_id=config.model_id,
            api_key=config.api_key,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            timeout=config.timeout,
        )

    else:
        raise ValueError(f"Unknown provider type: {config.provider_type}")


def get_provider(tier: ModelTier) -> ModelProvider:
    """
    Get a provider for the specified tier.

    This is the main entry point for getting a configured provider.

    Args:
        tier: Model tier (TIER1 for fast/cheap, TIER2 for quality).

    Returns:
        Configured ModelProvider instance.

    Example:
        >>> from providers import get_provider, ModelTier
        >>> provider = get_provider(ModelTier.TIER1)
        >>> result = provider.complete(request)
    """
    config = get_config()

    if tier == ModelTier.TIER1:
        return create_provider(config.tier1_config)
    else:
        return create_provider(config.tier2_config)


def get_model_info(model_id: str) -> ModelInfo:
    """
    Get model info for any known model.

    Args:
        model_id: Model identifier.

    Returns:
        ModelInfo with pricing and capabilities.
    """
    from providers.anthropic import ANTHROPIC_MODELS
    from providers.gemini import GEMINI_MODELS
    from providers.openrouter import OPENROUTER_MODELS

    # Check each registry
    if model_id in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[model_id]
    if model_id in GEMINI_MODELS:
        return GEMINI_MODELS[model_id]
    if model_id in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model_id]

    # Unknown model - create placeholder
    from decimal import Decimal
    from providers.base import ModelPricing

    return ModelInfo(
        id=model_id,
        provider="unknown",
        name=model_id,
        context_window=32_000,
        pricing=ModelPricing(
            input_price=Decimal("1.00"),
            output_price=Decimal("2.00"),
        ),
    )
