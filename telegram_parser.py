#!/usr/bin/env python3
"""
Telegram Channel Parser
Exports messages from public Telegram channels to CSV and JSONL formats.

Usage Examples:
---------------

# Basic export (text only)
python telegram_parser.py @bbcnews

# Export with media download
python telegram_parser.py @bbcnews --with-media

# Export with CSV splitting for LLM analysis (recommended for large channels)
python telegram_parser.py @channel_name --split-size 500

# Custom output directory
python telegram_parser.py @channel_name --output ./custom_dir

# Full example with all options
python telegram_parser.py @channel_name \
    --with-media \
    --split-size 1000 \
    --output ./export_dir \
    --page-size 50

# Resume interrupted export (just run the same command)
python telegram_parser.py @channel_name --split-size 500

Parameters:
-----------
  channel               Channel to export (@username or t.me/slug)
  -o, --output DIR      Output directory (default: ./export)
  --with-media         Download media files (photos, documents)
  --page-size N        Messages per batch (default: 100)
  --split-size N       Split CSV into files with max N messages each
  --session NAME       Session file name (default: telegram_session)

Setup:
------
1. Install dependencies: pip install -r requirements.txt
2. Get API credentials from https://my.telegram.org
3. Create .env file with TG_API_ID and TG_API_HASH
4. Run the script - it will prompt for phone/password on first use

Notes:
------
- Script automatically resumes from last message on interruption
- CSV splitting creates numbered files (messages_001.csv, messages_002.csv, etc.)
- JSONL remains single file for resume tracking
- Handles rate limits and 2FA authentication automatically
"""

import os
import json
import csv
import re
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Tuple

from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto
from tqdm import tqdm
from dotenv import load_dotenv


def normalize_channel(channel: str) -> str:
    """Normalize channel input to @username format."""
    m = re.match(r"^(https?://t\.me/)?(@?[\w\d_]+)$", channel.strip())
    if not m:
        raise ValueError("Provide @channel_username or t.me/slug format.")
    slug = m.group(2)
    return slug if slug.startswith("@") else f"@{slug}"


def get_last_message_id(jsonl_path: Path) -> int:
    """Get the last message ID from existing JSONL file for resuming."""
    last_id = 0
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    last_id = max(last_id, obj.get("id", 0))
                except Exception:
                    continue
    return last_id


def init_csv(csv_path: Path) -> bool:
    """Initialize CSV file with headers if it doesn't exist."""
    csv_exists = csv_path.exists()
    if not csv_exists:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(
                [
                    "id",
                    "date_iso",
                    "from_id",
                    "from_name",
                    "text",
                    "views",
                    "forwards",
                    "replies",
                    "media_type",
                    "media_path",
                    "link",
                    "reply_to_msg_id",
                    "grouped_id",
                ]
            )
    return csv_exists


async def download_media(
    message, media_dir: Path
) -> Tuple[Optional[str], Optional[str]]:
    """Download media from message if available."""
    if not message.media:
        return None, None

    if not (
        isinstance(message.media, MessageMediaPhoto)
        or isinstance(message.media, MessageMediaDocument)
    ):
        return None, None

    try:
        fname = await message.download_media(file=str(media_dir))
        media_type = type(message.media).__name__
        return media_type, fname
    except Exception as e:
        return "download_error", str(e)


class CSVWriter:
    """Handle CSV writing with optional splitting support."""

    def __init__(self, output_dir: Path, split_size: Optional[int] = None):
        self.output_dir = output_dir
        self.split_size = split_size
        self.current_file = None
        self.current_writer = None
        self.current_file_count = 0
        self.current_file_index = 1
        self.csv_files = []

        if split_size:
            self._find_latest_split_file()
        else:
            # Traditional single file approach
            csv_path = output_dir / "messages.csv"
            csv_exists = csv_path.exists()
            self.csv_files.append(csv_path)

            if not csv_exists:
                self.current_file = open(csv_path, "w", newline="", encoding="utf-8")
                self.current_writer = csv.writer(self.current_file)
                self._write_headers()
            else:
                self.current_file = open(csv_path, "a", newline="", encoding="utf-8")
                self.current_writer = csv.writer(self.current_file)

    def _find_latest_split_file(self):
        """Find the latest split file and prepare for appending or create new one."""
        import glob

        # Find existing split files
        pattern = str(self.output_dir / "messages_*.csv")
        existing_files = sorted(glob.glob(pattern))

        if existing_files:
            # Find the highest numbered file
            latest_file = existing_files[-1]
            latest_path = Path(latest_file)

            # Extract file number from filename
            filename = latest_path.stem
            file_num_str = filename.split("_")[-1]
            try:
                self.current_file_index = int(file_num_str)
            except ValueError:
                self.current_file_index = 1

            # Add all existing files to the list
            for file_path in existing_files:
                self.csv_files.append(Path(file_path))

            # Count rows in the latest file to see if it's full
            self.current_file_count = (
                self._count_rows_in_file(latest_path) - 1
            )  # Subtract header

            if self.current_file_count >= self.split_size:
                # Latest file is full, create a new one
                self.current_file_index += 1
                self._open_new_file()
            else:
                # Latest file has space, append to it
                self.current_file = open(latest_path, "a", newline="", encoding="utf-8")
                self.current_writer = csv.writer(self.current_file)
        else:
            # No existing files, create the first one
            self._open_new_file()

    def _count_rows_in_file(self, file_path: Path) -> int:
        """Count the number of rows in a CSV file."""
        if not file_path.exists():
            return 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _write_headers(self):
        """Write CSV headers."""
        self.current_writer.writerow(
            [
                "id",
                "date_iso",
                "from_id",
                "from_name",
                "text",
                "views",
                "forwards",
                "replies",
                "media_type",
                "media_path",
                "link",
                "reply_to_msg_id",
                "grouped_id",
            ]
        )

    def _open_new_file(self):
        """Open a new split CSV file."""
        if self.current_file:
            self.current_file.close()

        csv_path = self.output_dir / f"messages_{self.current_file_index:03d}.csv"
        self.csv_files.append(csv_path)
        self.current_file = open(csv_path, "w", newline="", encoding="utf-8")
        self.current_writer = csv.writer(self.current_file)
        self._write_headers()
        self.current_file_count = 0

    def writerow(self, row):
        """Write a row, potentially opening a new file if split size reached."""
        if self.split_size and self.current_file_count >= self.split_size:
            self.current_file_index += 1
            self._open_new_file()

        self.current_writer.writerow(row)
        if self.split_size:
            self.current_file_count += 1

    def close(self):
        """Close the current file."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None


def extract_message_data(
    message, entity, media_type: Optional[str], media_path: Optional[str]
) -> dict:
    """Extract relevant data from a Telegram message."""
    # Extract reply_to_msg_id if message is a reply
    reply_to_msg_id: Optional[int] = None
    if message.reply_to is not None:
        reply_to_msg_id = getattr(message.reply_to, "reply_to_msg_id", None)

    # Extract grouped_id for album messages
    grouped_id: Optional[int] = getattr(message, "grouped_id", None)

    row = {
        "id": message.id,
        "date_iso": message.date.isoformat() if message.date else None,
        "from_id": getattr(getattr(message, "from_id", None), "user_id", None)
        or getattr(message, "sender_id", None),
        "from_name": None,
        "text": message.message
        or (message.raw_text if hasattr(message, "raw_text") else None),
        "views": getattr(message, "views", None),
        "forwards": getattr(message, "forwards", None),
        "replies": getattr(getattr(message, "replies", None), "replies", None),
        "media_type": media_type,
        "media_path": media_path,
        "link": f"https://t.me/{entity.username}/{message.id}"
        if getattr(entity, "username", None)
        else None,
        "reply_to_msg_id": reply_to_msg_id,
        "grouped_id": grouped_id,
    }

    # Try to resolve sender name
    try:
        if message.sender:
            row["from_name"] = (
                getattr(message.sender, "username", None)
                or getattr(message.sender, "title", None)
                or f"{getattr(message.sender, 'first_name', '')} {getattr(message.sender, 'last_name', '')}".strip()
                or None
            )
    except Exception:
        pass

    return row


async def export_channel(
    client: TelegramClient,
    channel: str,
    output_dir: Path,
    download_media: bool = False,
    page_size: int = 100,
    split_size: Optional[int] = None,
) -> int:
    """Export messages from a Telegram channel."""
    # Setup paths
    jsonl_path = output_dir / "messages.jsonl"
    media_dir = output_dir / "media"

    # Normalize channel name
    channel = normalize_channel(channel)

    # Join channel if needed
    try:
        await client(JoinChannelRequest(channel))
    except Exception:
        pass  # Might already be joined or not needed for public channels

    entity = await client.get_entity(channel)

    # Prepare for resuming
    last_id = get_last_message_id(jsonl_path)

    # Initialize CSV writer with optional splitting
    csv_writer_manager = CSVWriter(output_dir, split_size)

    if download_media:
        media_dir.mkdir(parents=True, exist_ok=True)

    total_fetched = 0
    offset_id = last_id
    pbar = tqdm(desc="Fetching messages", unit="msg")

    while True:
        batch = []
        async for msg in client.iter_messages(
            entity, limit=page_size, min_id=offset_id, reverse=True
        ):
            batch.append(msg)

        if not batch:
            break

        # Process and write batch
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            for msg in batch:
                # Download media if requested
                media_type, media_path = None, None
                if download_media:
                    media_type, media_path = await download_media(msg, media_dir)

                # Extract message data
                row = extract_message_data(msg, entity, media_type, media_path)

                # Write to files
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                csv_writer_manager.writerow(
                    [
                        row["id"],
                        row["date_iso"],
                        row["from_id"],
                        row["from_name"],
                        (row["text"] or "").replace("\n", "\\n"),
                        row["views"],
                        row["forwards"],
                        row["replies"],
                        row["media_type"],
                        row["media_path"],
                        row["link"],
                        row["reply_to_msg_id"],
                        row["grouped_id"],
                    ]
                )

                total_fetched += 1
                offset_id = max(offset_id, msg.id)
                pbar.update(1)

    pbar.close()
    csv_writer_manager.close()

    print(f"\n✓ Export complete!")
    print(f"  New messages fetched: {total_fetched}")

    # Show all CSV files created
    if len(csv_writer_manager.csv_files) == 1:
        print(f"  CSV:   {csv_writer_manager.csv_files[0].absolute()}")
    else:
        print(f"  CSV files ({len(csv_writer_manager.csv_files)} total):")
        for csv_file in csv_writer_manager.csv_files:
            print(f"    {csv_file.absolute()}")

    print(f"  JSONL: {jsonl_path.absolute()}")
    if download_media and media_dir.exists():
        print(f"  Media: {media_dir.absolute()}")

    return total_fetched


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export Telegram channel messages to CSV/JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s @bbcnews                     # Export text only
  %(prog)s @bbcnews --with-media        # Also download media
  %(prog)s t.me/bbcnews --output ./bbc  # Custom output directory
  %(prog)s @channel --page-size 50      # Smaller batches
  %(prog)s @channel --split-size 1000   # Split CSV into files with 1000 messages each
""",
    )

    parser.add_argument("channel", help="Channel to export (@username or t.me/slug)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./export"),
        help="Output directory (default: ./export)",
    )
    parser.add_argument(
        "--with-media",
        action="store_true",
        help="Download media files (photos, documents)",
    )
    parser.add_argument(
        "--page-size", type=int, default=100, help="Messages per batch (default: 100)"
    )
    parser.add_argument(
        "--session",
        default="telegram_session",
        help="Session file name (default: telegram_session)",
    )
    parser.add_argument(
        "--split-size",
        type=int,
        help="Split CSV into multiple files with max N messages each (e.g., --split-size 1000)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get API credentials from environment
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")

    if not api_id or not api_hash:
        print("Error: TG_API_ID and TG_API_HASH must be set in .env file")
        print("\nTo get your API credentials:")
        print("1. Go to https://my.telegram.org")
        print("2. Log in with your phone number")
        print("3. Go to 'API development tools'")
        print("4. Create an app to get API_ID and API_HASH")
        print("5. Add them to your .env file")
        return 1

    try:
        api_id = int(api_id)
    except ValueError:
        print("Error: TG_API_ID must be a number")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize client and run export
    client = TelegramClient(str(args.output / args.session), api_id, api_hash)

    try:
        await client.start()
        await export_channel(
            client,
            args.channel,
            args.output,
            download_media=args.with_media,
            page_size=args.page_size,
            split_size=args.split_size,
        )
    except KeyboardInterrupt:
        print("\n\nExport interrupted. You can resume by running the same command.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        await client.disconnect()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

