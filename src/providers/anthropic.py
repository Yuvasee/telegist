#!/usr/bin/env python3
"""
Anthropic Claude API provider implementation.

Direct access to Claude models for highest quality synthesis.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import tiktoken

from providers.base import (
    AuthenticationError,
    BaseProvider,
    CompletionRequest,
    CompletionResult,
    Message,
    MessageRole,
    ModelInfo,
    ModelPricing,
    ProviderError,
    RateLimitError,
    TokenUsage,
)


# =============================================================================
# Model Registry
# =============================================================================

ANTHROPIC_MODELS: dict[str, ModelInfo] = {
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        name="Claude 3.5 Sonnet",
        context_window=200_000,
        pricing=ModelPricing(
            input_price=Decimal("3.00"),
            output_price=Decimal("15.00"),
        ),
    ),
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        provider="anthropic",
        name="Claude Sonnet 4",
        context_window=200_000,
        pricing=ModelPricing(
            input_price=Decimal("3.00"),
            output_price=Decimal("15.00"),
        ),
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        id="claude-3-5-haiku-20241022",
        provider="anthropic",
        name="Claude 3.5 Haiku",
        context_window=200_000,
        pricing=ModelPricing(
            input_price=Decimal("0.80"),
            output_price=Decimal("4.00"),
        ),
    ),
    "claude-3-opus-20240229": ModelInfo(
        id="claude-3-opus-20240229",
        provider="anthropic",
        name="Claude 3 Opus",
        context_window=200_000,
        pricing=ModelPricing(
            input_price=Decimal("15.00"),
            output_price=Decimal("75.00"),
        ),
    ),
}


# =============================================================================
# Anthropic Provider
# =============================================================================


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude API provider.

    Uses the anthropic library for direct API access.
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
        Initialize Anthropic provider.

        Args:
            model_id: Claude model ID (e.g., "claude-3-5-sonnet-20241022").
            api_key: Anthropic API key.
            max_retries: Maximum retry attempts.
            retry_delay: Base delay between retries.
            timeout: Request timeout in seconds.
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
        )

        # Lazy import to avoid dependency if not using Anthropic
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install anthropic"
            ) from e

        # Initialize client
        self._client = self._anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout,
        )

        # Initialize tokenizer (Claude uses similar tokenization to cl100k_base)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_info(self) -> ModelInfo:
        """Get model info, with fallback for unknown models."""
        if self._model_id in ANTHROPIC_MODELS:
            return ANTHROPIC_MODELS[self._model_id]

        # Fallback for unknown models
        return ModelInfo(
            id=self._model_id,
            provider="anthropic",
            name=self._model_id,
            context_window=200_000,
            pricing=ModelPricing(
                input_price=Decimal("3.00"),
                output_price=Decimal("15.00"),
            ),
        )

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """
        Execute a completion request via Anthropic API.
        """
        start_time = time.time()

        # Extract system message and convert other messages
        system_content, messages = self._convert_messages(request.messages)

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                # Build request kwargs
                kwargs: dict[str, Any] = {
                    "model": self._model_id,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens or 4096,  # Anthropic requires max_tokens
                }

                if system_content:
                    kwargs["system"] = system_content

                if request.stop_sequences:
                    kwargs["stop_sequences"] = request.stop_sequences

                response = self._client.messages.create(**kwargs)

                return self._parse_response(response, start_time, request.request_id)

            except self._anthropic.RateLimitError:
                last_error = RateLimitError(self.name)
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))

            except self._anthropic.AuthenticationError as e:
                raise AuthenticationError(self.name, str(e))

            except self._anthropic.BadRequestError as e:
                raise ProviderError(f"Bad request: {e}", self.name)

            except self._anthropic.APIError as e:
                last_error = ProviderError(f"API error: {e}", self.name)
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))

            except Exception as e:
                last_error = ProviderError(f"Unexpected error: {e}", self.name)
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))

        raise last_error or ProviderError("Unknown error after retries", self.name)

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        Convert messages to Anthropic format.

        Anthropic handles system prompts separately from the message list.

        Returns:
            Tuple of (system_content, messages_list)
        """
        system_content: str | None = None
        converted: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic takes system as a separate parameter
                system_content = msg.content
            else:
                converted.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return system_content, converted

    def _parse_response(
        self,
        response: Any,
        start_time: float,
        request_id: str | None,
    ) -> CompletionResult:
        """Parse Anthropic API response."""
        # Extract content
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

        # Extract usage
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        # Calculate cost
        cost = self._calculate_cost(usage)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        return CompletionResult(
            content=content,
            usage=usage,
            model=response.model,
            cost=cost,
            finish_reason=response.stop_reason,
            request_id=request_id,
            latency_ms=latency_ms,
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken as approximation.

        Note: Claude uses its own tokenizer which may differ slightly.
        For precise counting, use Anthropic's count_tokens API.
        """
        if not text:
            return 0
        return len(self._tokenizer.encode(text))
