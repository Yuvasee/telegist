#!/usr/bin/env python3
"""
Semantic Chunker for Telegram Message Exports

Intelligently chunks JSONL message exports based on semantic boundaries:
- Temporal gaps between messages
- Reply chain grouping (thread reconstruction)
- Album/grouped message preservation
- Token-aware sizing with configurable overlap

Usage Examples:
---------------

# Basic chunking with defaults (20 min gap, 800 tokens, 15% overlap)
python semantic_chunker.py export/messages.jsonl

# Custom temporal gap threshold (30 minutes)
python semantic_chunker.py export/messages.jsonl --gap-minutes 30

# Larger chunks for context-heavy analyses
python semantic_chunker.py export/messages.jsonl --max-tokens 2000

# Output to specific directory with custom overlap
python semantic_chunker.py export/messages.jsonl \
    --output-dir chunks/ \
    --overlap 0.2

# Single file output (all chunks in one JSONL)
python semantic_chunker.py export/messages.jsonl --single-file

# Verbose mode with chunk statistics
python semantic_chunker.py export/messages.jsonl --verbose

Parameters:
-----------
  input_file            JSONL file from telegram_parser.py
  -o, --output-dir      Output directory for chunks (default: ./chunks)
  --gap-minutes N       Temporal gap threshold in minutes (default: 20)
  --max-tokens N        Maximum tokens per chunk (default: 800)
  --overlap FLOAT       Overlap ratio between chunks (default: 0.15)
  --encoding NAME       Tiktoken encoding name (default: cl100k_base)
  --format FORMAT       Output format: jsonl or json (default: jsonl)
  --single-file         Output all chunks to a single file
  -v, --verbose         Print detailed chunk statistics

Notes:
------
- Preserves reply chains and grouped messages (albums) together
- Uses tiktoken cl100k_base encoding (GPT-4/Claude compatible)
- Generates metadata for each chunk (thread_id, time_range, token_count)
- Overlap ensures context continuity between chunks
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import tiktoken


# =============================================================================
# Data Models
# =============================================================================


@dataclass(frozen=True)
class Message:
    """Immutable representation of a Telegram message."""

    id: int
    date_iso: str
    text: str
    reply_to_msg_id: int | None = None
    grouped_id: int | None = None
    from_id: int | None = None
    from_name: str | None = None
    views: int | None = None
    forwards: int | None = None
    replies: int | None = None
    media_type: str | None = None
    media_path: str | None = None
    link: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create Message from JSONL dict, handling missing/null fields."""
        # Extract known fields
        known_fields = {
            "id", "date_iso", "text", "reply_to_msg_id", "grouped_id",
            "from_id", "from_name", "views", "forwards", "replies",
            "media_type", "media_path", "link"
        }

        return cls(
            id=data.get("id", 0),
            date_iso=data.get("date_iso", ""),
            text=data.get("text") or "",
            reply_to_msg_id=data.get("reply_to_msg_id"),
            grouped_id=data.get("grouped_id"),
            from_id=data.get("from_id"),
            from_name=data.get("from_name"),
            views=data.get("views"),
            forwards=data.get("forwards"),
            replies=data.get("replies"),
            media_type=data.get("media_type"),
            media_path=data.get("media_path"),
            link=data.get("link"),
            # Store all original data for future field compatibility
            raw={k: v for k, v in data.items() if k not in known_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert back to dict for serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "date_iso": self.date_iso,
            "text": self.text,
            "reply_to_msg_id": self.reply_to_msg_id,
            "grouped_id": self.grouped_id,
            "from_id": self.from_id,
            "from_name": self.from_name,
            "views": self.views,
            "forwards": self.forwards,
            "replies": self.replies,
            "media_type": self.media_type,
            "media_path": self.media_path,
            "link": self.link,
        }
        # Merge raw fields back (for any unknown fields from original data)
        result.update(self.raw)
        return result

    @property
    def timestamp(self) -> datetime | None:
        """Parse ISO date string to datetime."""
        if not self.date_iso:
            return None
        try:
            # Handle ISO format with timezone
            return datetime.fromisoformat(self.date_iso.replace("Z", "+00:00"))
        except ValueError:
            return None


@dataclass
class ChunkMetadata:
    """Metadata describing a semantic chunk."""

    chunk_id: str
    chunk_index: int
    thread_ids: list[int]  # Message IDs that start reply threads in this chunk
    time_range_start: str
    time_range_end: str
    token_count: int
    message_count: int
    has_overlap: bool = False
    overlap_message_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "thread_ids": self.thread_ids,
            "time_range": {
                "start": self.time_range_start,
                "end": self.time_range_end,
            },
            "token_count": self.token_count,
            "message_count": self.message_count,
            "has_overlap": self.has_overlap,
            "overlap_message_ids": self.overlap_message_ids,
        }


@dataclass
class Chunk:
    """A semantic chunk of messages with metadata."""

    messages: list[Message]
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to dict."""
        return {
            "metadata": self.metadata.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
        }

    def to_text(self) -> str:
        """Convert messages to plain text for token counting."""
        parts: list[str] = []
        for msg in self.messages:
            if msg.text:
                # Unescape newlines that were escaped in CSV export
                text = msg.text.replace("\\n", "\n")
                parts.append(text)
        return "\n\n".join(parts)


# =============================================================================
# Boundary Detection
# =============================================================================


class BoundaryDetector(Protocol):
    """Protocol for semantic boundary detection strategies."""

    def is_boundary(self, prev_msg: Message, curr_msg: Message) -> bool:
        """Check if there's a semantic boundary between two messages."""
        ...


@dataclass
class TemporalBoundaryDetector:
    """Detects boundaries based on temporal gaps between messages."""

    gap_threshold: timedelta

    def is_boundary(self, prev_msg: Message, curr_msg: Message) -> bool:
        """Return True if time gap exceeds threshold."""
        prev_ts = prev_msg.timestamp
        curr_ts = curr_msg.timestamp

        if prev_ts is None or curr_ts is None:
            return False

        # Ensure we handle chronological ordering correctly
        gap = abs(curr_ts - prev_ts)
        return gap >= self.gap_threshold


@dataclass
class ReplyChainDetector:
    """Detects when a message is NOT part of an ongoing reply chain."""

    # Set of message IDs that have been seen (for reply chain tracking)
    seen_message_ids: set[int] = field(default_factory=set)

    def is_boundary(self, prev_msg: Message, curr_msg: Message) -> bool:
        """
        Return True if current message starts a new conversation thread.

        A message is NOT a boundary if:
        - It replies to a message in the current chunk
        - It shares grouped_id with the previous message (album)
        """
        # Never break up albums (grouped messages)
        if (prev_msg.grouped_id is not None
            and curr_msg.grouped_id is not None
            and prev_msg.grouped_id == curr_msg.grouped_id):
            return False

        # If current message replies to something we've seen, it's continuous
        if curr_msg.reply_to_msg_id and curr_msg.reply_to_msg_id in self.seen_message_ids:
            return False

        return True

    def track_message(self, msg: Message) -> None:
        """Add message ID to the seen set for reply chain tracking."""
        self.seen_message_ids.add(msg.id)

    def reset(self) -> None:
        """Clear tracking for a new chunk."""
        self.seen_message_ids.clear()


# =============================================================================
# Token Counter
# =============================================================================


class TokenCounter:
    """Counts tokens using tiktoken encoding."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_message(self, msg: Message) -> int:
        """Count tokens in a message's text content."""
        if not msg.text:
            return 0
        # Unescape newlines for accurate counting
        text = msg.text.replace("\\n", "\n")
        return self.count(text)

    def count_messages(self, messages: Sequence[Message]) -> int:
        """Count total tokens across multiple messages."""
        total = 0
        for msg in messages:
            total += self.count_message(msg)
            # Add separator tokens (approximate)
            total += 2  # For "\n\n" between messages
        return total


# =============================================================================
# JSONL Reader
# =============================================================================


def read_messages_from_jsonl(path: Path) -> Iterator[Message]:
    """
    Stream messages from JSONL file.

    Yields messages in file order (typically chronological from telegram_parser.py).
    Handles malformed lines gracefully.
    """
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield Message.from_dict(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}",
                      file=sys.stderr)
                continue


# =============================================================================
# Semantic Chunker
# =============================================================================


@dataclass
class ChunkerConfig:
    """Configuration for the semantic chunker."""

    gap_minutes: int = 20
    max_tokens: int = 800  # Optimized for chat data (500-1000 range)
    overlap_ratio: float = 0.15
    encoding_name: str = "cl100k_base"

    @property
    def gap_threshold(self) -> timedelta:
        return timedelta(minutes=self.gap_minutes)

    @property
    def overlap_tokens(self) -> int:
        return int(self.max_tokens * self.overlap_ratio)


class SemanticChunker:
    """
    Chunks messages based on semantic boundaries with token limits.

    Algorithm:
    1. Load all messages (needed for reply chain analysis)
    2. Build message index for reply chain lookups
    3. Identify semantic segments using boundary detectors
    4. Split segments into token-limited chunks with overlap
    """

    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.token_counter = TokenCounter(config.encoding_name)
        self.temporal_detector = TemporalBoundaryDetector(config.gap_threshold)

    def chunk(self, messages: Sequence[Message]) -> list[Chunk]:
        """
        Chunk messages into semantic groups.

        Returns list of Chunk objects with metadata.
        """
        if not messages:
            return []

        # Identify semantic segments first
        segments = self._identify_segments(messages)

        # Split segments into token-limited chunks with overlap
        chunks = self._split_segments_into_chunks(segments)

        return chunks

    def _identify_segments(
        self,
        messages: Sequence[Message],
    ) -> list[list[Message]]:
        """
        Group messages into semantic segments based on boundaries.

        A new segment starts when:
        - There's a temporal gap exceeding threshold AND
        - The message doesn't reply to something in the current segment AND
        - The message isn't part of an album with the previous message
        """
        if not messages:
            return []

        segments: list[list[Message]] = []
        current_segment: list[Message] = []
        reply_detector = ReplyChainDetector()

        for i, msg in enumerate(messages):
            if i == 0:
                # First message starts first segment
                current_segment.append(msg)
                reply_detector.track_message(msg)
                continue

            prev_msg = messages[i - 1]

            # Check for temporal boundary
            has_temporal_gap = self.temporal_detector.is_boundary(prev_msg, msg)

            # Check if this breaks a reply chain
            breaks_chain = reply_detector.is_boundary(prev_msg, msg)

            # Start new segment only if BOTH conditions are true
            # This ensures we don't break reply chains that span temporal gaps
            if has_temporal_gap and breaks_chain:
                if current_segment:
                    segments.append(current_segment)
                current_segment = [msg]
                reply_detector.reset()
                reply_detector.track_message(msg)
            else:
                current_segment.append(msg)
                reply_detector.track_message(msg)

        # Don't forget the last segment
        if current_segment:
            segments.append(current_segment)

        return segments

    def _split_segments_into_chunks(
        self,
        segments: list[list[Message]]
    ) -> list[Chunk]:
        """
        Split segments into token-limited chunks with overlap.

        If a segment exceeds max_tokens, it's split with overlap.
        If a segment fits within max_tokens, it becomes one chunk.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for segment in segments:
            segment_tokens = self.token_counter.count_messages(segment)

            if segment_tokens <= self.config.max_tokens:
                # Segment fits in one chunk
                chunk = self._create_chunk(segment, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Need to split segment with overlap
                sub_chunks = self._split_segment_with_overlap(segment, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        return chunks

    def _split_segment_with_overlap(
        self,
        segment: list[Message],
        start_index: int
    ) -> list[Chunk]:
        """
        Split a large segment into overlapping chunks.

        Uses a sliding window approach with configurable overlap.
        """
        chunks: list[Chunk] = []
        chunk_index = start_index

        i = 0
        while i < len(segment):
            chunk_messages: list[Message] = []
            chunk_tokens = 0

            # Collect messages until we hit max tokens
            while i < len(segment):
                msg = segment[i]
                msg_tokens = self.token_counter.count_message(msg)

                # Always include at least one message per chunk
                if chunk_messages and (chunk_tokens + msg_tokens) > self.config.max_tokens:
                    break

                chunk_messages.append(msg)
                chunk_tokens += msg_tokens + 2  # +2 for separator
                i += 1

            # Create chunk with overlap info
            overlap_ids: list[int] = []
            has_overlap = False

            if chunks:
                # Add overlap from previous chunk
                prev_chunk = chunks[-1]
                overlap_messages = self._get_overlap_messages(prev_chunk.messages)
                if overlap_messages:
                    has_overlap = True
                    overlap_ids = [m.id for m in overlap_messages]

            chunk = self._create_chunk(
                chunk_messages,
                chunk_index,
                has_overlap=has_overlap,
                overlap_message_ids=overlap_ids
            )
            chunks.append(chunk)
            chunk_index += 1

            # If we haven't processed all messages, back up by overlap amount
            if i < len(segment):
                overlap_count = self._calculate_overlap_message_count(chunk_messages)
                i = max(i - overlap_count, i - len(chunk_messages) + 1)

        return chunks

    def _get_overlap_messages(self, messages: list[Message]) -> list[Message]:
        """Get messages from end of chunk that should overlap into next chunk."""
        if not messages:
            return []

        overlap_tokens = 0
        overlap_messages: list[Message] = []

        # Work backwards from end
        for msg in reversed(messages):
            msg_tokens = self.token_counter.count_message(msg)
            if overlap_tokens + msg_tokens > self.config.overlap_tokens:
                break
            overlap_messages.insert(0, msg)
            overlap_tokens += msg_tokens

        return overlap_messages

    def _calculate_overlap_message_count(self, messages: list[Message]) -> int:
        """Calculate how many messages from end should overlap."""
        return len(self._get_overlap_messages(messages))

    def _create_chunk(
        self,
        messages: list[Message],
        chunk_index: int,
        has_overlap: bool = False,
        overlap_message_ids: list[int] | None = None
    ) -> Chunk:
        """Create a Chunk with computed metadata."""
        if not messages:
            raise ValueError("Cannot create chunk with no messages")

        # Compute thread IDs - messages that are reply targets within this chunk
        message_ids = {m.id for m in messages}
        thread_ids: list[int] = []

        for msg in messages:
            if msg.reply_to_msg_id and msg.reply_to_msg_id in message_ids:
                if msg.reply_to_msg_id not in thread_ids:
                    thread_ids.append(msg.reply_to_msg_id)

        # Get time range
        timestamps = [m.timestamp for m in messages if m.timestamp]
        if timestamps:
            time_start = min(timestamps).isoformat()
            time_end = max(timestamps).isoformat()
        else:
            time_start = messages[0].date_iso or ""
            time_end = messages[-1].date_iso or ""

        metadata = ChunkMetadata(
            chunk_id=str(uuid.uuid4())[:8],
            chunk_index=chunk_index,
            thread_ids=thread_ids,
            time_range_start=time_start,
            time_range_end=time_end,
            token_count=self.token_counter.count_messages(messages),
            message_count=len(messages),
            has_overlap=has_overlap,
            overlap_message_ids=overlap_message_ids or [],
        )

        return Chunk(messages=messages, metadata=metadata)


# =============================================================================
# Output Writers
# =============================================================================


def write_chunks_jsonl(chunks: list[Chunk], output_dir: Path, prefix: str = "chunk") -> list[Path]:
    """Write chunks to individual JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []

    for chunk in chunks:
        filename = f"{prefix}_{chunk.metadata.chunk_index:03d}.jsonl"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            # Write metadata as first line
            f.write(json.dumps({"_metadata": chunk.metadata.to_dict()}, ensure_ascii=False) + "\n")
            # Write each message
            for msg in chunk.messages:
                f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")

        created_files.append(output_path)

    return created_files


def write_chunks_json(chunks: list[Chunk], output_dir: Path, prefix: str = "chunk") -> list[Path]:
    """Write chunks to individual JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []

    for chunk in chunks:
        filename = f"{prefix}_{chunk.metadata.chunk_index:03d}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False)

        created_files.append(output_path)

    return created_files


def write_chunks_single_file(chunks: list[Chunk], output_dir: Path, prefix: str = "chunks") -> Path:
    """Write all chunks to a single JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    return output_path


def write_manifest(chunks: list[Chunk], output_dir: Path) -> Path:
    """Write a manifest file summarizing all chunks."""
    manifest: dict[str, Any] = {
        "total_chunks": len(chunks),
        "total_messages": sum(c.metadata.message_count for c in chunks),
        "total_tokens": sum(c.metadata.token_count for c in chunks),
        "chunks": [c.metadata.to_dict() for c in chunks],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


# =============================================================================
# CLI
# =============================================================================


def print_statistics(chunks: list[Chunk], verbose: bool = False) -> None:
    """Print chunk statistics to stdout."""
    total_messages = sum(c.metadata.message_count for c in chunks)
    total_tokens = sum(c.metadata.token_count for c in chunks)

    print(f"\nChunking Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total messages: {total_messages}")
    print(f"  Total tokens: {total_tokens:,}")

    if chunks:
        avg_tokens = total_tokens // len(chunks)
        avg_messages = total_messages // len(chunks)
        print(f"  Avg tokens/chunk: {avg_tokens:,}")
        print(f"  Avg messages/chunk: {avg_messages}")

    if verbose and chunks:
        print(f"\nChunk Details:")
        print("-" * 70)
        for chunk in chunks:
            m = chunk.metadata
            print(
                f"  [{m.chunk_index:03d}] "
                f"{m.message_count:4d} msgs, "
                f"{m.token_count:5,} tokens, "
                f"threads: {len(m.thread_ids)}, "
                f"overlap: {'yes' if m.has_overlap else 'no'}"
            )
            if m.time_range_start and m.time_range_end:
                print(f"         Time: {m.time_range_start[:19]} to {m.time_range_end[:19]}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantically chunk Telegram message exports for LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s export/messages.jsonl
  %(prog)s export/messages.jsonl --gap-minutes 30 --max-tokens 2000
  %(prog)s messages.jsonl -o chunks/ --overlap 0.2 -v
  %(prog)s messages.jsonl --format json --verbose
  %(prog)s messages.jsonl --single-file
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
        default=Path("./chunks"),
        help="Output directory for chunks (default: ./chunks)"
    )
    parser.add_argument(
        "--gap-minutes",
        type=int,
        default=20,
        help="Temporal gap threshold in minutes (default: 20)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum tokens per chunk (default: 800)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="Overlap ratio between chunks, 0.0-1.0 (default: 0.15)"
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="Tiktoken encoding name (default: cl100k_base)"
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format (default: jsonl)"
    )
    parser.add_argument(
        "--prefix",
        default="chunk",
        help="Output file prefix (default: chunk)"
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Output all chunks to a single JSONL file instead of separate files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed chunk statistics"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        return 1

    if not 0.0 <= args.overlap <= 1.0:
        print("Error: Overlap must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    if args.max_tokens < 100:
        print("Error: max-tokens must be at least 100", file=sys.stderr)
        return 1

    # Configure chunker
    config = ChunkerConfig(
        gap_minutes=args.gap_minutes,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap,
        encoding_name=args.encoding,
    )

    print("Semantic Chunker")
    print("================")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}/")
    print("Config:")
    print(f"  Gap threshold: {config.gap_minutes} minutes")
    print(f"  Max tokens: {config.max_tokens:,}")
    print(f"  Overlap: {config.overlap_ratio:.0%} ({config.overlap_tokens:,} tokens)")
    print(f"  Encoding: {config.encoding_name}")
    if args.single_file:
        print("  Mode: Single file output")

    # Load messages
    print("\nLoading messages...")
    messages = list(read_messages_from_jsonl(args.input_file))
    print(f"  Loaded {len(messages)} messages")

    if not messages:
        print("Error: No messages found in input file", file=sys.stderr)
        return 1

    # Chunk messages
    print("\nChunking...")
    chunker = SemanticChunker(config)
    chunks = chunker.chunk(messages)

    # Print statistics
    print_statistics(chunks, args.verbose)

    # Write output
    print("\nWriting chunks...")
    if args.single_file:
        output_path = write_chunks_single_file(chunks, args.output_dir, args.prefix)
        print(f"\nOutput:")
        print(f"  Created single file: {output_path}")
    else:
        if args.format == "jsonl":
            created_files = write_chunks_jsonl(chunks, args.output_dir, args.prefix)
        else:
            created_files = write_chunks_json(chunks, args.output_dir, args.prefix)
        print(f"\nOutput:")
        print(f"  Created {len(created_files)} chunk files in {args.output_dir}/")

    manifest_path = write_manifest(chunks, args.output_dir)
    print(f"  Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
