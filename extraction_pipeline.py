#!/usr/bin/env python3
"""
Two-Tier Extraction Pipeline

Orchestrates intelligent extraction from Telegram message exports:
1. Tier 1: Parallel chunk summarization using fast/cheap model
2. Tier 2: Final synthesis using quality model

Features:
- Configurable prompt templates
- Confidence scoring (1-5) for extracted items
- Confidence-weighted merging in synthesis
- Real-time progress reporting
- Comprehensive cost tracking
- Graceful error handling with partial results

Usage:
    python extraction_pipeline.py messages.jsonl -o output/
    python extraction_pipeline.py messages.jsonl --max-parallel 3 --show-cost
    python extraction_pipeline.py messages.jsonl --tier1-model openai/gpt-4o-mini
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from cost_tracker import CostTracker
from preprocessor import Preprocessor, PreprocessorConfig, read_messages_from_jsonl
from providers import CompletionRequest, ModelTier, get_provider, get_config, ProviderConfig
from providers.base import Message as ProviderMessage, MessageRole
from providers.config import create_provider, _detect_provider_type, _get_api_key_for_provider
from semantic_chunker import Chunk, ChunkerConfig, SemanticChunker, Message


# =============================================================================
# Prompt Templates
# =============================================================================

DEFAULT_TIER1_PROMPT = """You are an expert analyst extracting structured insights from Telegram channel messages.

## Messages to Analyze:
{messages}

## Task:
Extract the following categories with confidence scores (1-5, where 5 is highest confidence):

1. **Key Insights**: Main takeaways, observations, important information
2. **Action Items**: Tasks, recommendations, things to do
3. **Resources**: Links, tools, services, platforms mentioned
4. **Notable Quotes**: Memorable or significant statements worth preserving verbatim

## Output Format:
Return ONLY valid JSON in this exact structure:
{{
  "key_insights": [
    {{"text": "insight text", "confidence": 4}},
    ...
  ],
  "action_items": [
    {{"text": "action item", "confidence": 3}},
    ...
  ],
  "resources": [
    {{"url": "https://...", "description": "what it is", "confidence": 5}},
    ...
  ],
  "notable_quotes": [
    {{"quote": "exact quote text", "context": "brief context", "confidence": 4}},
    ...
  ]
}}

Important:
- Only include items actually present in the messages
- Use extractive quotes (exact text) for notable quotes
- Assign confidence based on clarity and importance
- Output valid JSON only, no markdown code blocks in the response
"""

DEFAULT_TIER2_PROMPT = """You are synthesizing multiple chunk summaries into a comprehensive final report.

## Chunk Summaries:
{summaries}

## Task:
Create a well-organized final report that:
1. Merges similar insights (weighted by confidence scores)
2. Deduplicates action items and resources
3. Preserves the most notable quotes
4. Organizes content by theme/topic

## Output Format:
Markdown with the following structure:

# Extraction Summary

## Key Insights
[Merged insights organized by theme, with high-confidence items prioritized]

## Action Items
[Deduplicated action items, sorted by confidence]

## Resources
[Unique resources with descriptions]

## Notable Quotes
[Best quotes with context]

## Statistics
- Total chunks processed: {chunk_count}
- Items extracted: [counts per category]

---
*Generated on {timestamp}*
"""


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ExtractedItem:
    """Base class for extracted items with confidence."""
    text: str
    confidence: int  # 1-5 scale
    source_chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "source_chunk_id": self.source_chunk_id,
        }


@dataclass
class ResourceItem:
    """Extracted resource/link."""
    url: str
    description: str
    confidence: int
    source_chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "description": self.description,
            "confidence": self.confidence,
            "source_chunk_id": self.source_chunk_id,
        }


@dataclass
class QuoteItem:
    """Extracted notable quote."""
    quote: str
    context: str
    confidence: int
    source_chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote": self.quote,
            "context": self.context,
            "confidence": self.confidence,
            "source_chunk_id": self.source_chunk_id,
        }


@dataclass
class ChunkSummary:
    """Structured summary of a single chunk."""
    chunk_id: str
    chunk_index: int
    time_range_start: str
    time_range_end: str
    message_count: int
    token_count: int

    key_insights: list[ExtractedItem] = field(default_factory=list)
    action_items: list[ExtractedItem] = field(default_factory=list)
    resources: list[ResourceItem] = field(default_factory=list)
    notable_quotes: list[QuoteItem] = field(default_factory=list)

    # Processing metadata
    success: bool = True
    error_message: str | None = None
    processing_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "time_range": {
                "start": self.time_range_start,
                "end": self.time_range_end,
            },
            "message_count": self.message_count,
            "token_count": self.token_count,
            "key_insights": [i.to_dict() for i in self.key_insights],
            "action_items": [i.to_dict() for i in self.action_items],
            "resources": [r.to_dict() for r in self.resources],
            "notable_quotes": [q.to_dict() for q in self.notable_quotes],
            "success": self.success,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
        }

    @property
    def item_count(self) -> int:
        return (len(self.key_insights) + len(self.action_items) +
                len(self.resources) + len(self.notable_quotes))


@dataclass
class PipelineResult:
    """Complete extraction pipeline result."""
    input_file: str
    timestamp: str

    # Processing stats
    total_messages: int
    total_chunks: int
    successful_chunks: int
    failed_chunks: int

    # Extracted data
    chunk_summaries: list[ChunkSummary]
    final_synthesis: str

    # Cost tracking
    total_cost_usd: Decimal
    cost_breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_file": self.input_file,
            "timestamp": self.timestamp,
            "stats": {
                "total_messages": self.total_messages,
                "total_chunks": self.total_chunks,
                "successful_chunks": self.successful_chunks,
                "failed_chunks": self.failed_chunks,
            },
            "chunk_summaries": [s.to_dict() for s in self.chunk_summaries],
            "final_synthesis": self.final_synthesis,
            "cost": {
                "total_usd": str(self.total_cost_usd),
                "breakdown": self.cost_breakdown,
            },
        }


# =============================================================================
# Progress Reporting
# =============================================================================


@dataclass
class ProgressUpdate:
    """Progress update for callbacks."""
    phase: str  # "preprocessing", "chunking", "tier1", "tier2"
    current: int
    total: int
    message: str
    chunk_id: str | None = None


ProgressCallback = Callable[[ProgressUpdate], None]


def default_progress_callback(update: ProgressUpdate) -> None:
    """Default progress reporter to stdout."""
    pct = (update.current / update.total * 100) if update.total > 0 else 0
    if update.chunk_id:
        print(f"  [{update.phase}] {update.current}/{update.total} ({pct:.0f}%) - {update.message} [{update.chunk_id}]")
    else:
        print(f"  [{update.phase}] {update.current}/{update.total} ({pct:.0f}%) - {update.message}")


# =============================================================================
# Response Parsing
# =============================================================================


def parse_tier1_response(content: str, chunk_id: str) -> ChunkSummary:
    """Parse Tier 1 JSON response into ChunkSummary."""
    # Try to extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)

    # Also try without code blocks
    content = content.strip()
    if not content.startswith('{') and '{' in content:
        # Extract first JSON object
        start = content.index('{')
        content = content[start:]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Return empty summary on parse failure
        return ChunkSummary(
            chunk_id=chunk_id,
            chunk_index=0,
            time_range_start="",
            time_range_end="",
            message_count=0,
            token_count=0,
            success=False,
            error_message="Failed to parse JSON response",
        )

    # Parse key insights
    key_insights: list[ExtractedItem] = []
    for item in data.get("key_insights", []):
        if isinstance(item, dict):
            key_insights.append(ExtractedItem(
                text=item.get("text", ""),
                confidence=item.get("confidence", 3),
                source_chunk_id=chunk_id,
            ))

    # Parse action items
    action_items: list[ExtractedItem] = []
    for item in data.get("action_items", []):
        if isinstance(item, dict):
            action_items.append(ExtractedItem(
                text=item.get("text", ""),
                confidence=item.get("confidence", 3),
                source_chunk_id=chunk_id,
            ))

    # Parse resources
    resources: list[ResourceItem] = []
    for item in data.get("resources", []):
        if isinstance(item, dict):
            resources.append(ResourceItem(
                url=item.get("url", ""),
                description=item.get("description", ""),
                confidence=item.get("confidence", 3),
                source_chunk_id=chunk_id,
            ))

    # Parse quotes
    notable_quotes: list[QuoteItem] = []
    for item in data.get("notable_quotes", []):
        if isinstance(item, dict):
            notable_quotes.append(QuoteItem(
                quote=item.get("quote", ""),
                context=item.get("context", ""),
                confidence=item.get("confidence", 3),
                source_chunk_id=chunk_id,
            ))

    return ChunkSummary(
        chunk_id=chunk_id,
        chunk_index=0,  # Will be set by caller
        time_range_start="",
        time_range_end="",
        message_count=0,
        token_count=0,
        key_insights=key_insights,
        action_items=action_items,
        resources=resources,
        notable_quotes=notable_quotes,
        success=True,
    )


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    max_parallel: int = 5
    tier1_prompt: str = DEFAULT_TIER1_PROMPT
    tier2_prompt: str = DEFAULT_TIER2_PROMPT

    # Chunking config
    chunk_max_tokens: int = 800
    chunk_gap_minutes: int = 20
    chunk_overlap: float = 0.15

    # Model overrides (None = use defaults from providers/config.py)
    tier1_model: str | None = None
    tier2_model: str | None = None


# =============================================================================
# Extraction Pipeline
# =============================================================================


class ExtractionPipeline:
    """
    Two-tier extraction pipeline for Telegram message exports.

    Tier 1: Parallel extraction using fast/cheap model
    Tier 2: Synthesis using quality model
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        cost_log_file: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback or default_progress_callback
        self.cost_tracker = CostTracker(log_file=cost_log_file)
        self._progress_lock = threading.Lock()

        # Initialize providers
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize Tier 1 and Tier 2 providers."""
        # Tier 1 provider
        if self.config.tier1_model:
            provider_type = _detect_provider_type(self.config.tier1_model)
            tier1_config = ProviderConfig(
                provider_type=provider_type,
                model_id=self.config.tier1_model,
                api_key=_get_api_key_for_provider(provider_type),
            )
            self.tier1_provider = create_provider(tier1_config)
        else:
            self.tier1_provider = get_provider(ModelTier.TIER1)

        # Tier 2 provider
        if self.config.tier2_model:
            provider_type = _detect_provider_type(self.config.tier2_model)
            tier2_config = ProviderConfig(
                provider_type=provider_type,
                model_id=self.config.tier2_model,
                api_key=_get_api_key_for_provider(provider_type),
            )
            self.tier2_provider = create_provider(tier2_config)
        else:
            self.tier2_provider = get_provider(ModelTier.TIER2)

    def _report_progress(self, update: ProgressUpdate) -> None:
        """Thread-safe progress reporting."""
        with self._progress_lock:
            self.progress_callback(update)

    def run(
        self,
        input_file: Path,
        output_dir: Path | None = None,
        preprocess: bool = True,
    ) -> PipelineResult:
        """
        Run the full extraction pipeline.

        Args:
            input_file: Path to JSONL file from telegram_parser.py
            output_dir: Optional output directory for results
            preprocess: Whether to run preprocessing (dedup + link extraction)

        Returns:
            PipelineResult with all extracted data and cost info.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Phase 1: Load messages
        self._report_progress(ProgressUpdate(
            phase="loading", current=0, total=1, message="Loading messages..."
        ))
        raw_messages = list(read_messages_from_jsonl(input_file))
        total_messages = len(raw_messages)
        self._report_progress(ProgressUpdate(
            phase="loading", current=1, total=1, message=f"Loaded {total_messages} messages"
        ))

        # Convert preprocessor Message to semantic_chunker Message
        messages: list[Message] = [
            Message.from_dict(m.to_dict()) for m in raw_messages
        ]

        # Phase 2: Optional preprocessing
        if preprocess:
            self._report_progress(ProgressUpdate(
                phase="preprocessing", current=0, total=1, message="Preprocessing..."
            ))
            preprocessor = Preprocessor(PreprocessorConfig())
            clean_preprocessor_messages, stats = preprocessor.process(iter(raw_messages))
            # Convert back to semantic_chunker Message
            messages = [Message.from_dict(m.to_dict()) for m in clean_preprocessor_messages]
            self._report_progress(ProgressUpdate(
                phase="preprocessing", current=1, total=1,
                message=f"Removed {stats.exact_duplicates_removed + stats.near_duplicates_removed} duplicates"
            ))

        # Phase 3: Chunking
        self._report_progress(ProgressUpdate(
            phase="chunking", current=0, total=1, message="Chunking messages..."
        ))
        chunker_config = ChunkerConfig(
            max_tokens=self.config.chunk_max_tokens,
            gap_minutes=self.config.chunk_gap_minutes,
            overlap_ratio=self.config.chunk_overlap,
        )
        chunker = SemanticChunker(chunker_config)
        chunks = chunker.chunk(messages)
        self._report_progress(ProgressUpdate(
            phase="chunking", current=1, total=1,
            message=f"Created {len(chunks)} chunks"
        ))

        # Phase 4: Tier 1 - Parallel chunk extraction
        summaries = self._extract_tier1(chunks)

        # Phase 5: Tier 2 - Synthesis
        synthesis = self._synthesize_tier2(summaries, len(chunks), timestamp)

        # Compile results
        successful = sum(1 for s in summaries if s.success)
        failed = len(summaries) - successful

        result = PipelineResult(
            input_file=str(input_file),
            timestamp=timestamp,
            total_messages=total_messages,
            total_chunks=len(chunks),
            successful_chunks=successful,
            failed_chunks=failed,
            chunk_summaries=summaries,
            final_synthesis=synthesis,
            total_cost_usd=self.cost_tracker.total_cost,
            cost_breakdown=self.cost_tracker.get_summary().to_dict(),
        )

        # Write outputs if output_dir specified
        if output_dir:
            self._write_outputs(result, output_dir)

        return result

    def _extract_tier1(self, chunks: list[Chunk]) -> list[ChunkSummary]:
        """Run Tier 1 parallel extraction on all chunks."""
        summaries: list[ChunkSummary] = []
        total = len(chunks)
        completed = 0

        self._report_progress(ProgressUpdate(
            phase="tier1", current=0, total=total, message="Starting parallel extraction..."
        ))

        with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
            futures = {
                executor.submit(self._process_single_chunk, chunk): chunk
                for chunk in chunks
            }

            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    summary = future.result()
                except Exception as e:
                    # Create error summary
                    summary = ChunkSummary(
                        chunk_id=chunk.metadata.chunk_id,
                        chunk_index=chunk.metadata.chunk_index,
                        time_range_start=chunk.metadata.time_range_start,
                        time_range_end=chunk.metadata.time_range_end,
                        message_count=chunk.metadata.message_count,
                        token_count=chunk.metadata.token_count,
                        success=False,
                        error_message=str(e),
                    )

                summaries.append(summary)
                completed += 1

                status = "✓" if summary.success else "✗"
                self._report_progress(ProgressUpdate(
                    phase="tier1", current=completed, total=total,
                    message=f"{status} Extracted {summary.item_count} items",
                    chunk_id=summary.chunk_id,
                ))

        # Sort by chunk index
        summaries.sort(key=lambda s: s.chunk_index)
        return summaries

    def _process_single_chunk(self, chunk: Chunk) -> ChunkSummary:
        """Process a single chunk through Tier 1."""
        start_time = time.time()

        # Build prompt
        messages_text = chunk.to_text()
        prompt = self.config.tier1_prompt.format(messages=messages_text)

        # Call provider
        request = CompletionRequest(
            messages=[ProviderMessage(MessageRole.USER, prompt)],
            temperature=0.3,
        )

        result = self.tier1_provider.complete(request)

        # Track cost
        self.cost_tracker.record(
            result,
            chunk_id=chunk.metadata.chunk_id,
            tier="tier1",
        )

        # Parse response
        summary = parse_tier1_response(result.content, chunk.metadata.chunk_id)

        # Fill in chunk metadata
        summary.chunk_index = chunk.metadata.chunk_index
        summary.time_range_start = chunk.metadata.time_range_start
        summary.time_range_end = chunk.metadata.time_range_end
        summary.message_count = chunk.metadata.message_count
        summary.token_count = chunk.metadata.token_count
        summary.processing_time_ms = int((time.time() - start_time) * 1000)

        return summary

    def _synthesize_tier2(
        self,
        summaries: list[ChunkSummary],
        chunk_count: int,
        timestamp: str,
    ) -> str:
        """Run Tier 2 synthesis on chunk summaries."""
        self._report_progress(ProgressUpdate(
            phase="tier2", current=0, total=1, message="Starting synthesis..."
        ))

        # Filter successful summaries
        successful_summaries = [s for s in summaries if s.success]

        if not successful_summaries:
            return "# Extraction Summary\n\nNo chunks were successfully processed."

        # Build summaries JSON for prompt
        summaries_data = [s.to_dict() for s in successful_summaries]
        summaries_text = json.dumps(summaries_data, indent=2, ensure_ascii=False)

        # Build prompt
        prompt = self.config.tier2_prompt.format(
            summaries=summaries_text,
            chunk_count=chunk_count,
            timestamp=timestamp,
        )

        # Call provider
        request = CompletionRequest(
            messages=[ProviderMessage(MessageRole.USER, prompt)],
            temperature=0.3,
        )

        result = self.tier2_provider.complete(request)

        # Track cost
        self.cost_tracker.record(result, tier="tier2")

        self._report_progress(ProgressUpdate(
            phase="tier2", current=1, total=1, message="Synthesis complete"
        ))

        return result.content

    def _write_outputs(self, result: PipelineResult, output_dir: Path) -> None:
        """Write pipeline outputs to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full result as JSON
        result_path = output_dir / "extraction_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        # Synthesis as Markdown
        synthesis_path = output_dir / "synthesis.md"
        with open(synthesis_path, "w", encoding="utf-8") as f:
            f.write(result.final_synthesis)

        # Cost summary
        cost_path = output_dir / "cost_summary.txt"
        with open(cost_path, "w", encoding="utf-8") as f:
            f.write(self.cost_tracker.format_summary())

        print(f"\nOutputs written to {output_dir}/")
        print(f"  - extraction_result.json")
        print(f"  - synthesis.md")
        print(f"  - cost_summary.txt")

    def get_cost_summary(self) -> str:
        """Get formatted cost summary."""
        return self.cost_tracker.format_summary()


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Tier Extraction Pipeline for Telegram message exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s messages.jsonl
  %(prog)s messages.jsonl -o output/
  %(prog)s messages.jsonl --max-parallel 3 --show-cost
  %(prog)s messages.jsonl --tier1-model openai/gpt-4o-mini
  %(prog)s messages.jsonl --no-preprocess
""",
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="JSONL file from telegram_parser.py"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum parallel Tier 1 extractions (default: 5)"
    )
    parser.add_argument(
        "--tier1-model",
        type=str,
        help="Override Tier 1 model (default: from TIER1_MODEL env or google/gemini-flash-1.5)"
    )
    parser.add_argument(
        "--tier2-model",
        type=str,
        help="Override Tier 2 model (default: from TIER2_MODEL env or anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=800,
        help="Target tokens per chunk (default: 800)"
    )
    parser.add_argument(
        "--gap-minutes",
        type=int,
        default=20,
        help="Temporal gap threshold for chunking (default: 20)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing (deduplication)"
    )
    parser.add_argument(
        "--show-cost",
        action="store_true",
        help="Show detailed cost summary at end"
    )
    parser.add_argument(
        "--cost-log",
        type=Path,
        help="Log each API call to JSONL file"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        return 1

    # Create config
    config = PipelineConfig(
        max_parallel=args.max_parallel,
        chunk_max_tokens=args.chunk_tokens,
        chunk_gap_minutes=args.gap_minutes,
        tier1_model=args.tier1_model,
        tier2_model=args.tier2_model,
    )

    # Progress callback
    progress_cb: ProgressCallback | None = None if args.quiet else default_progress_callback

    # Print header
    if not args.quiet:
        print("=" * 60)
        print("Two-Tier Extraction Pipeline")
        print("=" * 60)
        print(f"Input: {args.input_file}")
        if args.output_dir:
            print(f"Output: {args.output_dir}/")
        print("Config:")
        print(f"  Max parallel: {config.max_parallel}")
        print(f"  Chunk tokens: {config.chunk_max_tokens}")
        print(f"  Gap minutes: {config.chunk_gap_minutes}")
        print(f"  Preprocessing: {'disabled' if args.no_preprocess else 'enabled'}")
        print()

    # Run pipeline
    try:
        pipeline = ExtractionPipeline(
            config=config,
            cost_log_file=args.cost_log,
            progress_callback=progress_cb,
        )

        result = pipeline.run(
            input_file=args.input_file,
            output_dir=args.output_dir,
            preprocess=not args.no_preprocess,
        )

        # Print summary
        if not args.quiet:
            print()
            print("=" * 60)
            print("EXTRACTION COMPLETE")
            print("=" * 60)
            print(f"Total messages: {result.total_messages:,}")
            print(f"Total chunks: {result.total_chunks}")
            print(f"Successful: {result.successful_chunks}")
            print(f"Failed: {result.failed_chunks}")
            print(f"Total cost: ${result.total_cost_usd:.6f}")

        if args.show_cost:
            print()
            print(pipeline.get_cost_summary())

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
