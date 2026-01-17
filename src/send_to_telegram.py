#!/usr/bin/env python3
"""
Send a message or file to Telegram Saved Messages.
Uses existing session from the parser.
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
import os

from telethon import TelegramClient


async def send_to_saved_messages(message: str, session_dir: str = ".") -> None:
    """Send a message to Saved Messages (self)."""
    load_dotenv()

    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")

    if not api_id or not api_hash:
        raise ValueError("TG_API_ID and TG_API_HASH must be set in .env")

    session_path = Path(session_dir) / "telegram_session"

    client = TelegramClient(str(session_path), int(api_id), api_hash)

    await client.start()

    # Send to "me" = Saved Messages
    await client.send_message("me", message)
    print("Message sent to Saved Messages!")

    await client.disconnect()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python send_to_telegram.py <file_path> [session_dir]")
        print("  file_path: Path to file to send as message")
        print("  session_dir: Directory containing telegram_session (default: current)")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    session_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    content = file_path.read_text()

    # Telegram has a 4096 character limit per message
    MAX_LEN = 4000

    if len(content) > MAX_LEN:
        # Split into multiple messages
        parts = []
        current = ""
        for line in content.split("\n"):
            if len(current) + len(line) + 1 > MAX_LEN:
                parts.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            parts.append(current)

        print(f"Message will be split into {len(parts)} parts")

        async def send_parts() -> None:
            load_dotenv()
            api_id = os.getenv("TG_API_ID")
            api_hash = os.getenv("TG_API_HASH")
            session_path = Path(session_dir) / "telegram_session"
            client = TelegramClient(str(session_path), int(api_id), api_hash)
            await client.start()

            for i, part in enumerate(parts, 1):
                await client.send_message("me", part)
                print(f"Sent part {i}/{len(parts)}")

            await client.disconnect()

        asyncio.run(send_parts())
    else:
        asyncio.run(send_to_saved_messages(content, session_dir))


if __name__ == "__main__":
    main()
