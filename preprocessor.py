#!/usr/bin/env python3
"""
Telegram Message Preprocessor

Preprocesses JSONL message exports to remove duplicates and extract links:
- Exact duplicate detection (hash-based on message text)
- Near-duplicate detection (Jaccard similarity > 0.95 on word tokens)
- Link extraction and cataloging from message text
- Comprehensive preprocessing statistics

Usage Examples:
---------------

# Basic preprocessing (exact + near-duplicate removal, link extraction)
python preprocessor.py export/messages.jsonl

# Custom output directory
python preprocessor.py export/messages.jsonl --output-dir preprocessed/

# Adjust near-duplicate threshold (0.0-1.0, default 0.95)
python preprocessor.py export/messages.jsonl --similarity-threshold 0.90

# Keep exact duplicates but remove near-duplicates
python preprocessor.py export/messages.jsonl --keep-exact-duplicates

# Skip near-duplicate detection (faster, exact duplicates only)
python preprocessor.py export/messages.jsonl --skip-near-duplicates

# Verbose output with detailed statistics
python preprocessor.py export/messages.jsonl --verbose

Parameters:
-----------
  input_file                  JSONL file from telegram_parser.py
  -o, --output-dir            Output directory (default: ./preprocessed)
  --similarity-threshold N    Jaccard similarity threshold for near-duplicates (default: 0.95)
  --keep-exact-duplicates     Keep exact duplicates (only remove near-duplicates)
  --skip-near-duplicates      Skip near-duplicate detection (faster processing)
  -v, --verbose               Print detailed statistics

Output Files:
-------------
  messages_clean.jsonl        Deduplicated messages
  links_catalog.json          Extracted links with message IDs and timestamps
  preprocessing_stats.json    Detailed statistics about preprocessing

Notes:
------
- Exact duplicates are detected via SHA-256 hash of normalized message text
- Near-duplicates use Jaccard similarity on word tokens (no ML dependencies)
- Link extraction handles URLs in message text using regex patterns
- Original message order is preserved in output
- Empty messages (no text) are preserved but not used for duplicate detection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol


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
    def normalized_text(self) -> str:
        """Normalize text for duplicate detection (lowercase, whitespace normalized)."""
        if not self.text:
            return ""
        # Unescape newlines, normalize whitespace, lowercase
        text = self.text.replace("\\n", "\n")
        text = " ".join(text.split())
        return text.lower()


@dataclass
class LinkInfo:
    """Information about an extracted link."""

    url: str
    message_ids: list[int]
    first_seen: str  # ISO timestamp of first occurrence
    occurrence_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "url": self.url,
            "message_ids": self.message_ids,
            "first_seen": self.first_seen,
            "occurrence_count": self.occurrence_count,
        }


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing operation."""

    total_messages: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    messages_after_dedup: int
    unique_links_found: int
    total_link_occurrences: int
    empty_messages: int  # Messages with no text

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        total_removed = self.exact_duplicates_removed + self.near_duplicates_removed
        dedup_rate = f"{total_removed / self.total_messages * 100:.2f}%" if self.total_messages > 0 else "0.00%"

        return {
            "total_messages": self.total_messages,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
            "total_duplicates_removed": total_removed,
            "messages_after_dedup": self.messages_after_dedup,
            "unique_links_found": self.unique_links_found,
            "total_link_occurrences": self.total_link_occurrences,
            "empty_messages": self.empty_messages,
            "deduplication_rate": dedup_rate,
        }


# =============================================================================
# Duplicate Detection
# =============================================================================


class DuplicateDetector(Protocol):
    """Protocol for duplicate detection strategies."""

    def is_duplicate(self, msg: Message) -> bool:
        """Check if message is a duplicate of any seen message."""
        ...


class ExactDuplicateDetector:
    """Detects exact duplicates using SHA-256 hash of normalized text."""

    def __init__(self) -> None:
        self.seen_hashes: set[str] = set()

    def is_duplicate(self, msg: Message) -> bool:
        """
        Check if message text is an exact duplicate.

        Returns True if this exact text has been seen before.
        Empty messages are never considered duplicates.
        """
        if not msg.text or not msg.normalized_text:
            return False

        text_hash = self._hash_text(msg.normalized_text)
        if text_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(text_hash)
        return False

    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate SHA-256 hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NearDuplicateDetector:
    """
    Detects near-duplicates using Jaccard similarity on word tokens.

    Uses a sliding window approach to compare each message against
    recent messages (to avoid O(n^2) comparisons on large datasets).
    """

    def __init__(self, similarity_threshold: float = 0.95, window_size: int = 1000):
        """
        Initialize near-duplicate detector.

        Args:
            similarity_threshold: Jaccard similarity threshold (0.0-1.0)
            window_size: Number of recent messages to compare against
        """
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.recent_messages: list[Message] = []

    def is_duplicate(self, msg: Message) -> bool:
        """
        Check if message is a near-duplicate of any recent message.

        Returns True if Jaccard similarity with any recent message exceeds threshold.
        Empty messages are never considered duplicates.
        """
        if not msg.text or not msg.normalized_text:
            return False

        msg_tokens = self._tokenize(msg.normalized_text)
        if not msg_tokens:
            return False

        # Compare against recent messages
        for recent_msg in self.recent_messages:
            recent_tokens = self._tokenize(recent_msg.normalized_text)
            if not recent_tokens:
                continue

            similarity = self._jaccard_similarity(msg_tokens, recent_tokens)
            if similarity >= self.similarity_threshold:
                return True

        # Add to recent messages window
        self.recent_messages.append(msg)
        if len(self.recent_messages) > self.window_size:
            self.recent_messages.pop(0)

        return False

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into word tokens (lowercased, alphanumeric)."""
        # Split on whitespace and non-alphanumeric characters
        words = re.findall(r'\w+', text.lower())
        return set(words)

    @staticmethod
    def _jaccard_similarity(set1: set[str], set2: set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Jaccard similarity = |intersection| / |union|
        Returns value between 0.0 (no overlap) and 1.0 (identical).
        """
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union


# =============================================================================
# Link Extraction
# =============================================================================


class LinkExtractor:
    """Extracts and catalogs links from message text."""

    # Regex pattern for URLs (http/https)
    URL_PATTERN = re.compile(
        r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}'
        r'\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
        re.IGNORECASE
    )

    def __init__(self) -> None:
        # Map from URL to LinkInfo
        self.links: dict[str, LinkInfo] = {}

    def extract_from_message(self, msg: Message) -> list[str]:
        """
        Extract all URLs from message text.

        Returns list of extracted URLs.
        """
        if not msg.text:
            return []

        # Unescape newlines for proper URL detection
        text = msg.text.replace("\\n", "\n")
        urls = self.URL_PATTERN.findall(text)

        # Update link catalog
        for url in urls:
            if url not in self.links:
                self.links[url] = LinkInfo(
                    url=url,
                    message_ids=[msg.id],
                    first_seen=msg.date_iso,
                    occurrence_count=1,
                )
            else:
                link_info = self.links[url]
                # Update existing link info
                self.links[url] = LinkInfo(
                    url=link_info.url,
                    message_ids=link_info.message_ids + [msg.id],
                    first_seen=link_info.first_seen,
                    occurrence_count=link_info.occurrence_count + 1,
                )

        return urls

    def get_catalog(self) -> dict[str, Any]:
        """
        Get link catalog as dict.

        Returns dict mapping URLs to LinkInfo dicts.
        """
        return {
            url: info.to_dict()
            for url, info in self.links.items()
        }

    def get_statistics(self) -> tuple[int, int]:
        """
        Get link statistics.

        Returns (unique_links_count, total_occurrences).
        """
        unique_count = len(self.links)
        total_occurrences = sum(info.occurrence_count for info in self.links.values())
        return unique_count, total_occurrences


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
# Preprocessor
# =============================================================================


@dataclass
class PreprocessorConfig:
    """Configuration for the preprocessor."""

    similarity_threshold: float = 0.95
    near_duplicate_window: int = 1000
    skip_exact_duplicates: bool = False
    skip_near_duplicates: bool = False


class Preprocessor:
    """
    Main preprocessor for Telegram message exports.

    Performs:
    1. Exact duplicate detection and removal
    2. Near-duplicate detection and removal
    3. Link extraction and cataloging
    """

    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.exact_detector = ExactDuplicateDetector()
        self.near_detector = NearDuplicateDetector(
            similarity_threshold=config.similarity_threshold,
            window_size=config.near_duplicate_window,
        )
        self.link_extractor = LinkExtractor()

    def process(self, messages: Iterator[Message]) -> tuple[list[Message], PreprocessingStats]:
        """
        Process messages: remove duplicates and extract links.

        Returns (deduplicated_messages, statistics).
        """
        clean_messages: list[Message] = []
        total_count = 0
        exact_dup_count = 0
        near_dup_count = 0
        empty_count = 0

        for msg in messages:
            total_count += 1

            # Track empty messages
            if not msg.text or not msg.normalized_text:
                empty_count += 1
                # Keep empty messages in output
                clean_messages.append(msg)
                # Extract links even from messages without text (might have link field)
                self.link_extractor.extract_from_message(msg)
                continue

            # Check for exact duplicates
            if not self.config.skip_exact_duplicates:
                if self.exact_detector.is_duplicate(msg):
                    exact_dup_count += 1
                    continue

            # Check for near-duplicates
            if not self.config.skip_near_duplicates:
                if self.near_detector.is_duplicate(msg):
                    near_dup_count += 1
                    continue

            # Message is not a duplicate - keep it
            clean_messages.append(msg)

            # Extract links
            self.link_extractor.extract_from_message(msg)

        # Get link statistics
        unique_links, total_links = self.link_extractor.get_statistics()

        stats = PreprocessingStats(
            total_messages=total_count,
            exact_duplicates_removed=exact_dup_count,
            near_duplicates_removed=near_dup_count,
            messages_after_dedup=len(clean_messages),
            unique_links_found=unique_links,
            total_link_occurrences=total_links,
            empty_messages=empty_count,
        )

        return clean_messages, stats


# =============================================================================
# Output Writers
# =============================================================================


def write_messages_jsonl(messages: list[Message], output_path: Path) -> None:
    """Write messages to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")


def write_links_catalog(link_catalog: dict[str, Any], output_path: Path) -> None:
    """Write links catalog to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort links by occurrence count (descending) for readability
    sorted_catalog = dict(
        sorted(
            link_catalog.items(),
            key=lambda item: item[1]["occurrence_count"],
            reverse=True,
        )
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_unique_links": len(sorted_catalog),
                "links": sorted_catalog,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_statistics(stats: PreprocessingStats, output_path: Path) -> None:
    """Write preprocessing statistics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)


# =============================================================================
# CLI
# =============================================================================


def print_statistics(stats: PreprocessingStats, verbose: bool = False) -> None:
    """Print preprocessing statistics to stdout."""
    print("\nPreprocessing Statistics:")
    print("=" * 70)
    print(f"  Total messages:           {stats.total_messages:,}")
    print(f"  Exact duplicates removed: {stats.exact_duplicates_removed:,}")
    print(f"  Near duplicates removed:  {stats.near_duplicates_removed:,}")
    print(f"  Messages after dedup:     {stats.messages_after_dedup:,}")
    print(f"  Empty messages:           {stats.empty_messages:,}")

    total_removed = stats.exact_duplicates_removed + stats.near_duplicates_removed
    if stats.total_messages > 0:
        dedup_rate = total_removed / stats.total_messages * 100
        print(f"  Deduplication rate:       {dedup_rate:.2f}%")

    print(f"\nLink Extraction:")
    print(f"  Unique links found:       {stats.unique_links_found:,}")
    print(f"  Total link occurrences:   {stats.total_link_occurrences:,}")

    if verbose:
        print(f"\nDetailed Breakdown:")
        print("-" * 70)
        print(f"  Original message count:       {stats.total_messages:,}")
        print(f"  After exact dedup:            {stats.total_messages - stats.exact_duplicates_removed:,}")
        print(f"  After near dedup:             {stats.messages_after_dedup:,}")
        print(f"  Messages retained:            {stats.messages_after_dedup / stats.total_messages * 100:.2f}%")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess Telegram message exports (deduplication and link extraction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s export/messages.jsonl
  %(prog)s messages.jsonl --output-dir preprocessed/ --verbose
  %(prog)s messages.jsonl --similarity-threshold 0.90
  %(prog)s messages.jsonl --skip-near-duplicates
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
        default=Path("./preprocessed"),
        help="Output directory (default: ./preprocessed)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Jaccard similarity threshold for near-duplicates, 0.0-1.0 (default: 0.95)"
    )
    parser.add_argument(
        "--keep-exact-duplicates",
        action="store_true",
        help="Keep exact duplicates (only remove near-duplicates)"
    )
    parser.add_argument(
        "--skip-near-duplicates",
        action="store_true",
        help="Skip near-duplicate detection (faster, exact duplicates only)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed statistics"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        return 1

    if not 0.0 <= args.similarity_threshold <= 1.0:
        print("Error: similarity-threshold must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    # Configure preprocessor
    config = PreprocessorConfig(
        similarity_threshold=args.similarity_threshold,
        near_duplicate_window=1000,
        skip_exact_duplicates=args.keep_exact_duplicates,
        skip_near_duplicates=args.skip_near_duplicates,
    )

    print("Telegram Message Preprocessor")
    print("=" * 70)
    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_dir}/")
    print("\nConfiguration:")
    print(f"  Exact duplicate detection:  {'disabled' if config.skip_exact_duplicates else 'enabled'}")
    print(f"  Near duplicate detection:   {'disabled' if config.skip_near_duplicates else 'enabled'}")
    if not config.skip_near_duplicates:
        print(f"  Similarity threshold:       {config.similarity_threshold:.2f}")
        print(f"  Comparison window size:     {config.near_duplicate_window:,} messages")

    # Load messages
    print("\nLoading messages...")
    messages = read_messages_from_jsonl(args.input_file)

    # Process messages
    print("Processing...")
    preprocessor = Preprocessor(config)
    clean_messages, stats = preprocessor.process(messages)

    # Print statistics
    print_statistics(stats, args.verbose)

    # Write outputs
    print("\nWriting outputs...")

    output_messages = args.output_dir / "messages_clean.jsonl"
    write_messages_jsonl(clean_messages, output_messages)
    print(f"  Deduplicated messages: {output_messages}")

    output_links = args.output_dir / "links_catalog.json"
    write_links_catalog(preprocessor.link_extractor.get_catalog(), output_links)
    print(f"  Links catalog:         {output_links}")

    output_stats = args.output_dir / "preprocessing_stats.json"
    write_statistics(stats, output_stats)
    print(f"  Statistics:            {output_stats}")

    print("\nPreprocessing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
