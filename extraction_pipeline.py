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

DEFAULT_TIER1_PROMPT = """You are analyzing a Telegram chat discussion to identify interesting conversation threads.

## Messages to Analyze:
{messages}

## Task:
Identify distinct discussion threads/topics in these messages. For each thread, extract:
- The topic/subject being discussed
- Key participants (usernames)
- The flow of conversation (what was said, who responded)
- Notable quotes that capture the essence
- Any links/resources mentioned

## Language Rule:
- If messages are in Russian, write your output in Russian
- For other languages, write in English

## Output Format:
Return ONLY valid JSON:
{{
  "discussions": [
    {{
      "topic": "Brief topic title",
      "participants": ["username1", "username2"],
      "summary": "2-4 sentences describing what was discussed, the key points made, and any conclusions reached",
      "quotes": [
        {{"author": "username", "text": "exact quote in original language"}}
      ],
      "links": [
        {{"url": "https://...", "context": "why it was shared"}}
      ],
      "importance": 3
    }}
  ],
  "language": "ru" or "en"
}}

Important:
- Focus on substantive discussions, skip trivial greetings/small talk
- Preserve exact quotes in original language
- Importance: 1-5 scale (5 = very interesting/valuable discussion)
- Output valid JSON only, no markdown code blocks
"""

DEFAULT_TIER2_PROMPT = """You are creating a digest of interesting discussions from a Telegram chat.

## Extracted Discussions:
{summaries}

## Task:
Create a readable digest that tells the story of each interesting discussion. For each discussion:
1. Merge related topics from different chunks into cohesive narratives
2. Include participant names to show who said what
3. Embed relevant quotes naturally within the narrative
4. Include links where they add value
5. Skip low-importance discussions (importance < 3)

## Language Rule:
- Check the "language" field in summaries
- If language is "ru", write the ENTIRE digest in Russian
- Otherwise, write in English

## Output Format:
Markdown with this structure:

# Дайджест обсуждений (or "Discussion Digest" for English)

## [Topic Title]

[2-5 paragraph narrative describing the discussion flow. Who started it, what points were made,
how others responded. Embed quotes naturally like: @username отметил: "quote here".
Include links inline where relevant: обсуждали [тему](url).]

---

## [Next Topic Title]

[Another discussion narrative...]

---

**Период**: [start date] — [end date]
**Сообщений**: [total from {chunk_count} chunks]
"""


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DiscussionQuote:
    """A quote from a discussion participant."""
    author: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"author": self.author, "text": self.text}


@dataclass
class DiscussionLink:
    """A link shared in a discussion."""
    url: str
    context: str

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "context": self.context}


@dataclass
class Discussion:
    """A discussion thread extracted from chat messages."""
    topic: str
    participants: list[str]
    summary: str
    quotes: list[DiscussionQuote]
    links: list[DiscussionLink]
    importance: int  # 1-5 scale
    source_chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "participants": self.participants,
            "summary": self.summary,
            "quotes": [q.to_dict() for q in self.quotes],
            "links": [l.to_dict() for l in self.links],
            "importance": self.importance,
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

    discussions: list[Discussion] = field(default_factory=list)
    language: str = "en"  # detected language: "ru" or "en"

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
            "discussions": [d.to_dict() for d in self.discussions],
            "language": self.language,
            "success": self.success,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
        }

    @property
    def item_count(self) -> int:
        return len(self.discussions)


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

    # Parse discussions
    discussions: list[Discussion] = []
    for item in data.get("discussions", []):
        if isinstance(item, dict):
            # Parse quotes
            quotes: list[DiscussionQuote] = []
            for q in item.get("quotes", []):
                if isinstance(q, dict):
                    quotes.append(DiscussionQuote(
                        author=q.get("author", ""),
                        text=q.get("text", ""),
                    ))

            # Parse links
            links: list[DiscussionLink] = []
            for link in item.get("links", []):
                if isinstance(link, dict):
                    links.append(DiscussionLink(
                        url=link.get("url", ""),
                        context=link.get("context", ""),
                    ))

            discussions.append(Discussion(
                topic=item.get("topic", ""),
                participants=item.get("participants", []),
                summary=item.get("summary", ""),
                quotes=quotes,
                links=links,
                importance=item.get("importance", 3),
                source_chunk_id=chunk_id,
            ))

    # Get detected language
    language = data.get("language", "en")

    return ChunkSummary(
        chunk_id=chunk_id,
        chunk_index=0,  # Will be set by caller
        time_range_start="",
        time_range_end="",
        message_count=0,
        token_count=0,
        discussions=discussions,
        language=language,
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
