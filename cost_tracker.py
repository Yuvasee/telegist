#!/usr/bin/env python3
"""
Cost Tracker Module

Tracks tokens and costs across API calls for the extraction pipeline.
Supports per-model tracking, session totals, and optional file logging.

Usage:
    from cost_tracker import CostTracker

    tracker = CostTracker(log_file="costs.jsonl")

    # Track after each API call
    tracker.record(result)  # CompletionResult from provider

    # Get summary
    summary = tracker.get_summary()
    print(tracker.format_summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from providers.base import CompletionResult, TokenUsage


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ModelStats:
    """Accumulated statistics for a specific model."""

    model_id: str
    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))

    def record(self, usage: TokenUsage, cost: Decimal) -> None:
        """Record a single API call."""
        self.call_count += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost += cost

    def to_dict(self) -> dict[str, str | int]:
        """Serialize to dict."""
        return {
            "model_id": self.model_id,
            "call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": str(self.total_cost),
        }


@dataclass
class CostSummary:
    """Summary of all tracked costs."""

    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost: Decimal
    by_model: dict[str, ModelStats]
    start_time: datetime
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": str(self.total_cost),
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


# =============================================================================
# Cost Tracker
# =============================================================================


class CostTracker:
    """
    Tracks API costs across multiple calls.

    Accumulates token usage and costs per model, providing
    session totals and detailed breakdowns.

    Example:
        >>> tracker = CostTracker()
        >>> result = provider.complete(request)
        >>> tracker.record(result)
        >>> print(tracker.format_summary())
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        *,
        auto_flush: bool = True,
    ):
        """
        Initialize cost tracker.

        Args:
            log_file: Optional path to log each call as JSONL.
            auto_flush: If True, flush log file after each write.
        """
        self._start_time = datetime.now(timezone.utc)
        self._model_stats: dict[str, ModelStats] = {}
        self._log_file: TextIO | None = None
        self._log_path: Path | None = None
        self._auto_flush = auto_flush

        if log_file:
            self._log_path = Path(log_file)
            self._log_file = open(self._log_path, "a", encoding="utf-8")

    def __enter__(self) -> CostTracker:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - close log file."""
        self.close()

    def close(self) -> None:
        """Close the log file if open."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def record(
        self,
        result: CompletionResult,
        *,
        chunk_id: str | None = None,
        tier: str | None = None,
    ) -> None:
        """
        Record a completion result.

        Args:
            result: CompletionResult from a provider.
            chunk_id: Optional chunk identifier for tracking.
            tier: Optional tier label (e.g., "tier1", "tier2").
        """
        model_id = result.model

        # Get or create model stats
        if model_id not in self._model_stats:
            self._model_stats[model_id] = ModelStats(model_id=model_id)

        # Record the call
        self._model_stats[model_id].record(result.usage, result.cost)

        # Log to file if configured
        if self._log_file:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_id,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cost_usd": str(result.cost),
                "latency_ms": result.latency_ms,
            }
            if chunk_id:
                log_entry["chunk_id"] = chunk_id
            if tier:
                log_entry["tier"] = tier

            self._log_file.write(json.dumps(log_entry) + "\n")
            if self._auto_flush:
                self._log_file.flush()

    def get_summary(self) -> CostSummary:
        """
        Get current cost summary.

        Returns:
            CostSummary with totals and per-model breakdown.
        """
        total_calls = sum(s.call_count for s in self._model_stats.values())
        total_prompt = sum(s.prompt_tokens for s in self._model_stats.values())
        total_completion = sum(s.completion_tokens for s in self._model_stats.values())
        total_tokens = sum(s.total_tokens for s in self._model_stats.values())
        total_cost = sum(
            (s.total_cost for s in self._model_stats.values()),
            start=Decimal("0"),
        )

        return CostSummary(
            total_calls=total_calls,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            total_cost=total_cost,
            by_model=dict(self._model_stats),
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
        )

    def format_summary(self, *, include_model_breakdown: bool = True) -> str:
        """
        Format a human-readable cost summary.

        Args:
            include_model_breakdown: Include per-model details.

        Returns:
            Formatted summary string.
        """
        summary = self.get_summary()
        lines = [
            "=" * 60,
            "COST SUMMARY",
            "=" * 60,
            f"Total API Calls:       {summary.total_calls:,}",
            f"Total Prompt Tokens:   {summary.total_prompt_tokens:,}",
            f"Total Completion Tokens: {summary.total_completion_tokens:,}",
            f"Total Tokens:          {summary.total_tokens:,}",
            f"Total Cost:            ${summary.total_cost:.6f}",
            f"Duration:              {self._format_duration(summary)}",
        ]

        if summary.total_calls > 0:
            avg_cost = summary.total_cost / Decimal(summary.total_calls)
            avg_tokens = summary.total_tokens // summary.total_calls
            lines.extend([
                "",
                "AVERAGES",
                f"Avg Cost per Call:     ${avg_cost:.6f}",
                f"Avg Tokens per Call:   {avg_tokens:,}",
            ])

        if include_model_breakdown and self._model_stats:
            lines.extend([
                "",
                "BY MODEL",
                "-" * 60,
            ])
            for model_id, stats in sorted(self._model_stats.items()):
                lines.extend([
                    f"  {model_id}:",
                    f"    Calls:      {stats.call_count:,}",
                    f"    Tokens:     {stats.total_tokens:,} (prompt: {stats.prompt_tokens:,}, completion: {stats.completion_tokens:,})",
                    f"    Cost:       ${stats.total_cost:.6f}",
                ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_duration(self, summary: CostSummary) -> str:
        """Format the session duration."""
        if summary.end_time is None:
            return "ongoing"

        delta = summary.end_time - summary.start_time
        seconds = int(delta.total_seconds())

        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def get_cost_per_messages(self, message_count: int) -> Decimal | None:
        """
        Calculate cost per N messages (for comparison with target).

        Args:
            message_count: Number of messages to scale to.

        Returns:
            Estimated cost per message_count messages, or None if no data.
        """
        summary = self.get_summary()
        if summary.total_calls == 0:
            return None

        # This is a rough estimate - actual cost depends on message lengths
        return summary.total_cost

    def to_dict(self) -> dict[str, object]:
        """Serialize tracker state to dict."""
        return self.get_summary().to_dict()

    def save_summary(self, path: str | Path) -> None:
        """
        Save summary to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary().to_dict(), f, indent=2, default=str)

    @property
    def total_cost(self) -> Decimal:
        """Get current total cost."""
        return sum(
            (s.total_cost for s in self._model_stats.values()),
            start=Decimal("0"),
        )

    @property
    def total_tokens(self) -> int:
        """Get current total tokens."""
        return sum(s.total_tokens for s in self._model_stats.values())

    @property
    def total_calls(self) -> int:
        """Get current total API calls."""
        return sum(s.call_count for s in self._model_stats.values())


# =============================================================================
# CLI Interface
# =============================================================================


def main() -> None:
    """CLI for viewing cost logs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cost Tracker - View and analyze cost logs"
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to JSONL cost log file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return

    # Reconstruct tracker from log file
    tracker = CostTracker()

    with open(args.log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)

            # Create a minimal CompletionResult for recording
            from providers.base import TokenUsage

            usage = TokenUsage(
                prompt_tokens=entry["prompt_tokens"],
                completion_tokens=entry["completion_tokens"],
                total_tokens=entry["total_tokens"],
            )

            # Record directly to model stats
            model_id = entry["model"]
            if model_id not in tracker._model_stats:
                tracker._model_stats[model_id] = ModelStats(model_id=model_id)
            tracker._model_stats[model_id].record(
                usage,
                Decimal(entry["cost_usd"]),
            )

    if args.json:
        print(json.dumps(tracker.get_summary().to_dict(), indent=2, default=str))
    else:
        print(tracker.format_summary())


if __name__ == "__main__":
    main()
