#!/usr/bin/env python3
"""Find a Telegram chat by name."""

import asyncio
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient


async def find_chat(search: str, session_dir: str = ".") -> None:
    """Search for a chat by name."""
    load_dotenv()

    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")

    if not api_id or not api_hash:
        raise ValueError("TG_API_ID and TG_API_HASH must be set in .env")

    session_path = Path(session_dir) / "telegram_session"
    client = TelegramClient(str(session_path), int(api_id), api_hash)

    await client.start()

    print(f"Searching for: {search}\n")

    async for dialog in client.iter_dialogs():
        name = dialog.name or ""
        if search.lower() in name.lower():
            entity = dialog.entity
            entity_type = type(entity).__name__

            # Get username if available
            username = getattr(entity, 'username', None)
            username_str = f"@{username}" if username else "no username"

            print(f"Found: {name}")
            print(f"  Type: {entity_type}")
            print(f"  ID: {dialog.id}")
            print(f"  Username: {username_str}")
            print()

    await client.disconnect()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python find_chat.py <search_term> [session_dir]")
        sys.exit(1)

    search = sys.argv[1]
    session_dir = sys.argv[2] if len(sys.argv) > 2 else "radio_t_output"

    asyncio.run(find_chat(search, session_dir))


if __name__ == "__main__":
    main()
