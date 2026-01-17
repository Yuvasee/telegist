#!/usr/bin/env python3
"""
Google Gemini API provider implementation.

Direct access to Gemini models without going through OpenRouter,
useful when OpenRouter has issues or for cost optimization.
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

GEMINI_MODELS: dict[str, ModelInfo] = {
    "gemini-1.5-flash": ModelInfo(
        id="gemini-1.5-flash",
        provider="gemini",
        name="Gemini 1.5 Flash",
        context_window=1_000_000,
        pricing=ModelPricing(
            input_price=Decimal("0.075"),
            output_price=Decimal("0.30"),
        ),
    ),
    "gemini-1.5-flash-8b": ModelInfo(
        id="gemini-1.5-flash-8b",
        provider="gemini",
        name="Gemini 1.5 Flash 8B",
        context_window=1_000_000,
        pricing=ModelPricing(
            input_price=Decimal("0.0375"),
            output_price=Decimal("0.15"),
        ),
    ),
    "gemini-1.5-pro": ModelInfo(
        id="gemini-1.5-pro",
        provider="gemini",
        name="Gemini 1.5 Pro",
        context_window=2_000_000,
        pricing=ModelPricing(
            input_price=Decimal("1.25"),
            output_price=Decimal("5.00"),
        ),
    ),
    "gemini-2.0-flash-exp": ModelInfo(
        id="gemini-2.0-flash-exp",
        provider="gemini",
        name="Gemini 2.0 Flash (Experimental)",
        context_window=1_000_000,
        pricing=ModelPricing(
            input_price=Decimal("0.075"),  # Estimated
            output_price=Decimal("0.30"),
        ),
    ),
}


# =============================================================================
# Gemini Provider
# =============================================================================


class GeminiProvider(BaseProvider):
    """
    Google Gemini API provider.

    Uses the google-generativeai library for direct API access.
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
        Initialize Gemini provider.

        Args:
            model_id: Gemini model ID (e.g., "gemini-1.5-flash").
            api_key: Google AI API key.
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

        # Lazy import to avoid dependency if not using Gemini
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai package required for Gemini provider. "
                "Install with: pip install google-generativeai"
            ) from e

        # Configure the client
        self._genai.configure(api_key=api_key)

        # Initialize model
        self._client = self._genai.GenerativeModel(model_id)

        # Initialize tokenizer (approximate, Gemini uses its own tokenizer)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_info(self) -> ModelInfo:
        """Get model info, with fallback for unknown models."""
        if self._model_id in GEMINI_MODELS:
            return GEMINI_MODELS[self._model_id]

        # Fallback for unknown models
        return ModelInfo(
            id=self._model_id,
            provider="gemini",
            name=self._model_id,
            context_window=32_000,
            pricing=ModelPricing(
                input_price=Decimal("0.50"),
                output_price=Decimal("1.50"),
            ),
        )

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """
        Execute a completion request via Gemini API.
        """
        start_time = time.time()

        # Convert messages to Gemini format
        # Gemini uses a different conversation format
        contents = self._convert_messages(request.messages)

        # Configure generation
        generation_config = self._genai.types.GenerationConfig(
            temperature=request.temperature,
        )
        if request.max_tokens is not None:
            generation_config.max_output_tokens = request.max_tokens
        if request.stop_sequences:
            generation_config.stop_sequences = request.stop_sequences

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.generate_content(
                    contents,
                    generation_config=generation_config,
                )

                return self._parse_response(response, start_time, request)

            except self._genai.types.BlockedPromptException as e:
                raise ProviderError(
                    f"Prompt blocked by safety filters: {e}",
                    self.name,
                )

            except self._genai.types.StopCandidateException as e:
                # Content was generated but stopped early
                raise ProviderError(
                    f"Generation stopped: {e}",
                    self.name,
                )

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limiting
                if "quota" in error_str or "rate" in error_str:
                    last_error = RateLimitError(self.name)
                    if attempt < self._max_retries - 1:
                        time.sleep(self._retry_delay * (2 ** attempt))
                    continue

                # Check for auth errors
                if "api key" in error_str or "authentication" in error_str:
                    raise AuthenticationError(self.name, str(e))

                last_error = ProviderError(f"API error: {e}", self.name)
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))

        raise last_error or ProviderError("Unknown error after retries", self.name)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Convert messages to Gemini format.

        Gemini uses 'user' and 'model' roles, and handles system
        prompts differently.
        """
        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Gemini handles system prompts as model configuration
                # We'll prepend it to the first user message
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                content = msg.content
                if system_instruction and not contents:
                    # Prepend system instruction to first user message
                    content = f"{system_instruction}\n\n{content}"
                    system_instruction = None
                contents.append({"role": "user", "parts": [content]})
            elif msg.role == MessageRole.ASSISTANT:
                contents.append({"role": "model", "parts": [msg.content]})

        return contents

    def _parse_response(
        self,
        response: Any,
        start_time: float,
        request: CompletionRequest,
    ) -> CompletionResult:
        """Parse Gemini API response."""
        # Extract content
        content = ""
        finish_reason = None

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                content = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
            finish_reason = str(candidate.finish_reason) if hasattr(candidate, "finish_reason") else None

        # Extract usage (Gemini provides this in usage_metadata)
        prompt_tokens = 0
        completion_tokens = 0

        if hasattr(response, "usage_metadata"):
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
        else:
            # Estimate if not provided
            prompt_tokens = self.count_messages_tokens(request.messages)
            completion_tokens = self.count_tokens(content)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        # Calculate cost
        cost = self._calculate_cost(usage)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        return CompletionResult(
            content=content,
            usage=usage,
            model=self._model_id,
            cost=cost,
            finish_reason=finish_reason,
            request_id=request.request_id,
            latency_ms=latency_ms,
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken as approximation.

        Note: Gemini uses its own tokenizer which may differ slightly.
        For precise counting, use the Gemini count_tokens API.
        """
        if not text:
            return 0
        return len(self._tokenizer.encode(text))
