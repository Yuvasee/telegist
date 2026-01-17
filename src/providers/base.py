#!/usr/bin/env python3
"""
Base classes and protocols for model providers.

Defines the abstract interface that all providers must implement,
along with common data structures for requests, responses, and errors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Exceptions
# =============================================================================


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, provider: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class RateLimitError(ProviderError):
    """Raised when API rate limit is hit."""

    def __init__(self, provider: str, retry_after: float | None = None):
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, provider, {"retry_after": retry_after})
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""

    def __init__(self, provider: str, message: str = "Authentication failed"):
        super().__init__(message, provider)


class ModelNotFoundError(ProviderError):
    """Raised when requested model is not available."""

    def __init__(self, provider: str, model: str):
        super().__init__(f"Model '{model}' not found", provider, {"model": model})
        self.model = model


# =============================================================================
# Data Models
# =============================================================================


class MessageRole(str, Enum):
    """Message role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to API format."""
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass(frozen=True)
class TokenUsage:
    """Token usage from a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, int]:
        """Serialize to dict."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ModelPricing:
    """Pricing per 1M tokens (in USD)."""

    input_price: Decimal  # Price per 1M input tokens
    output_price: Decimal  # Price per 1M output tokens

    def calculate_cost(self, usage: TokenUsage) -> Decimal:
        """Calculate cost for given token usage."""
        input_cost = (Decimal(usage.prompt_tokens) / Decimal(1_000_000)) * self.input_price
        output_cost = (Decimal(usage.completion_tokens) / Decimal(1_000_000)) * self.output_price
        return input_cost + output_cost

    def to_dict(self) -> dict[str, str]:
        """Serialize to dict (as strings for precision)."""
        return {
            "input_price_per_1m": str(self.input_price),
            "output_price_per_1m": str(self.output_price),
        }


@dataclass(frozen=True)
class ModelInfo:
    """Information about a model."""

    id: str  # Model identifier (e.g., "gpt-4o-mini")
    provider: str  # Provider name (e.g., "openrouter", "gemini")
    name: str  # Human-readable name
    context_window: int  # Maximum context length in tokens
    pricing: ModelPricing
    supports_system_message: bool = True
    supports_streaming: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "provider": self.provider,
            "name": self.name,
            "context_window": self.context_window,
            "pricing": self.pricing.to_dict(),
            "supports_system_message": self.supports_system_message,
            "supports_streaming": self.supports_streaming,
        }


@dataclass
class CompletionRequest:
    """Request for a completion."""

    messages: list[Message]
    temperature: float = 0.3
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None

    # Optional metadata for tracking
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        result: dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.stop_sequences:
            result["stop"] = self.stop_sequences
        return result


@dataclass
class CompletionResult:
    """Result from a completion request."""

    content: str
    usage: TokenUsage
    model: str  # Actual model used (may differ from requested)
    cost: Decimal  # Cost in USD

    # Optional metadata
    finish_reason: str | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    raw_response: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        result: dict[str, Any] = {
            "content": self.content,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "cost_usd": str(self.cost),
        }
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        if self.request_id:
            result["request_id"] = self.request_id
        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms
        return result


# =============================================================================
# Provider Protocol
# =============================================================================


@runtime_checkable
class ModelProvider(Protocol):
    """
    Protocol defining the interface for model providers.

    All providers must implement these methods to be compatible
    with the extraction pipeline.
    """

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openrouter', 'gemini', 'anthropic')."""
        ...

    @property
    def model_id(self) -> str:
        """Current model identifier."""
        ...

    @property
    def model_info(self) -> ModelInfo:
        """Information about the current model."""
        ...

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """
        Execute a completion request.

        Args:
            request: The completion request with messages and parameters.

        Returns:
            CompletionResult with content, usage, and cost.

        Raises:
            ProviderError: On API errors.
            RateLimitError: When rate limited.
            AuthenticationError: On auth failure.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using this provider's tokenizer.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        ...

    def count_messages_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens in a list of messages.

        Includes any overhead from message formatting.

        Args:
            messages: List of messages to count.

        Returns:
            Total token count including formatting overhead.
        """
        ...


# =============================================================================
# Base Implementation
# =============================================================================


class BaseProvider(ABC):
    """
    Abstract base class for model providers.

    Provides common functionality and enforces the ModelProvider protocol.
    Subclasses must implement the abstract methods.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
    ):
        """
        Initialize the provider.

        Args:
            model_id: Model identifier to use.
            api_key: API key for authentication.
            max_retries: Maximum retry attempts on failure.
            retry_delay: Base delay between retries (exponential backoff).
            timeout: Request timeout in seconds.
        """
        self._model_id = model_id
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._model_info: ModelInfo | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    def model_id(self) -> str:
        """Current model identifier."""
        return self._model_id

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Information about the current model."""
        ...

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Execute a completion request."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def count_messages_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens in a list of messages.

        Default implementation sums individual message tokens plus overhead.
        Subclasses may override for more accurate counting.
        """
        total = 0
        for msg in messages:
            # Base token count for content
            total += self.count_tokens(msg.content)
            # Add overhead for role and message structure (~4 tokens)
            total += 4
        # Add overhead for assistant reply priming (~3 tokens)
        total += 3
        return total

    def _calculate_cost(self, usage: TokenUsage) -> Decimal:
        """Calculate cost from token usage."""
        return self.model_info.pricing.calculate_cost(usage)
