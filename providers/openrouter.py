#!/usr/bin/env python3
"""
OpenRouter API provider implementation.

OpenRouter provides access to multiple models through a unified API,
making it ideal for the two-tier extraction strategy.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import requests
import tiktoken

from providers.base import (
    AuthenticationError,
    BaseProvider,
    CompletionRequest,
    CompletionResult,
    Message,
    ModelInfo,
    ModelPricing,
    ProviderError,
    RateLimitError,
    TokenUsage,
)


# =============================================================================
# Model Registry
# =============================================================================

# Pricing per 1M tokens (USD) - Updated January 2026
# Source: https://openrouter.ai/docs#models
OPENROUTER_MODELS: dict[str, ModelInfo] = {
    # Tier 1 - Fast/Cheap models for chunk extraction
    "google/gemini-flash-1.5": ModelInfo(
        id="google/gemini-flash-1.5",
        provider="openrouter",
        name="Gemini 1.5 Flash",
        context_window=1_000_000,
        pricing=ModelPricing(
            input_price=Decimal("0.075"),
            output_price=Decimal("0.30"),
        ),
    ),
    "openai/gpt-4o-mini": ModelInfo(
        id="openai/gpt-4o-mini",
        provider="openrouter",
        name="GPT-4o Mini",
        context_window=128_000,
        pricing=ModelPricing(
            input_price=Decimal("0.15"),
            output_price=Decimal("0.60"),
        ),
    ),
    "qwen/qwen-2.5-72b-instruct": ModelInfo(
        id="qwen/qwen-2.5-72b-instruct",
        provider="openrouter",
        name="Qwen 2.5 72B Instruct",
        context_window=131_072,
        pricing=ModelPricing(
            input_price=Decimal("0.35"),
            output_price=Decimal("0.40"),
        ),
    ),
    "qwen/qwen3-235b-a22b-2507": ModelInfo(
        id="qwen/qwen3-235b-a22b-2507",
        provider="openrouter",
        name="Qwen3 235B A22B",
        context_window=131_072,
        pricing=ModelPricing(
            input_price=Decimal("0.14"),
            output_price=Decimal("0.14"),
        ),
    ),

    # Tier 2 - Quality models for synthesis
    "anthropic/claude-sonnet-4": ModelInfo(
        id="anthropic/claude-sonnet-4",
        provider="openrouter",
        name="Claude Sonnet 4",
        context_window=200_000,
        pricing=ModelPricing(
            input_price=Decimal("3.00"),
            output_price=Decimal("15.00"),
        ),
    ),
    "anthropic/claude-3.5-sonnet": ModelInfo(
        id="anthropic/claude-3.5-sonnet",
        provider="openrouter",
        name="Claude 3.5 Sonnet",
        context_window=200_000,
        pricing=ModelPricing(
            input_price=Decimal("3.00"),
            output_price=Decimal("15.00"),
        ),
    ),
    "openai/gpt-4o": ModelInfo(
        id="openai/gpt-4o",
        provider="openrouter",
        name="GPT-4o",
        context_window=128_000,
        pricing=ModelPricing(
            input_price=Decimal("2.50"),
            output_price=Decimal("10.00"),
        ),
    ),
}


# =============================================================================
# OpenRouter Provider
# =============================================================================


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter API provider.

    Provides access to multiple models through OpenRouter's unified API.
    Supports automatic retries with exponential backoff.
    """

    DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        api_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
    ):
        """
        Initialize OpenRouter provider.

        Args:
            model_id: OpenRouter model ID (e.g., "google/gemini-flash-1.5").
            api_key: OpenRouter API key.
            api_url: Optional custom API URL.
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
        self._api_url = api_url or self.DEFAULT_API_URL

        # Initialize tokenizer (cl100k_base is compatible with most models)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def model_info(self) -> ModelInfo:
        """Get model info, with fallback for unknown models."""
        if self._model_id in OPENROUTER_MODELS:
            return OPENROUTER_MODELS[self._model_id]

        # Fallback for unknown models - use conservative pricing
        return ModelInfo(
            id=self._model_id,
            provider="openrouter",
            name=self._model_id,
            context_window=32_000,
            pricing=ModelPricing(
                input_price=Decimal("1.00"),
                output_price=Decimal("2.00"),
            ),
        )

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """
        Execute a completion request via OpenRouter.

        Implements retry logic with exponential backoff.
        """
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/telegram-channel-parser",
            "X-Title": "Telegram Channel Parser",
        }

        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    self._api_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    retry_seconds = float(retry_after) if retry_after else None
                    raise RateLimitError(self.name, retry_seconds)

                # Handle authentication errors
                if response.status_code == 401:
                    raise AuthenticationError(self.name, "Invalid API key")

                # Handle other errors
                if response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    raise ProviderError(
                        f"API error {response.status_code}: {response.text[:200]}",
                        self.name,
                        {"status_code": response.status_code, "response": error_data},
                    )

                # Parse response
                data = response.json()
                return self._parse_response(data, start_time, request.request_id)

            except RateLimitError as e:
                wait_time = e.retry_after or (self._retry_delay * (2 ** attempt))
                if attempt < self._max_retries - 1:
                    time.sleep(wait_time)
                last_error = e

            except requests.exceptions.Timeout:
                last_error = ProviderError("Request timeout", self.name)
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))

            except requests.exceptions.RequestException as e:
                last_error = ProviderError(f"Request error: {e}", self.name)
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))

        # All retries exhausted
        raise last_error or ProviderError("Unknown error after retries", self.name)

    def _parse_response(
        self,
        data: dict[str, Any],
        start_time: float,
        request_id: str | None,
    ) -> CompletionResult:
        """Parse OpenRouter API response."""
        # Extract content
        choices = data.get("choices", [])
        if not choices:
            raise ProviderError("No choices in response", self.name, {"response": data})

        content = choices[0].get("message", {}).get("content", "")
        finish_reason = choices[0].get("finish_reason")

        # Extract usage
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Calculate cost
        cost = self._calculate_cost(usage)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        return CompletionResult(
            content=content,
            usage=usage,
            model=data.get("model", self._model_id),
            cost=cost,
            finish_reason=finish_reason,
            request_id=request_id,
            latency_ms=latency_ms,
            raw_response=data,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    def count_messages_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens in messages with OpenAI-compatible overhead.

        Based on OpenAI's token counting guidelines.
        """
        total = 0
        for msg in messages:
            # Every message follows <|start|>{role}\n{content}<|end|>\n
            total += 4  # Overhead per message
            total += self.count_tokens(msg.content)
            total += self.count_tokens(msg.role.value)
        # Every reply is primed with <|start|>assistant<|message|>
        total += 3
        return total
