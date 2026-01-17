#!/usr/bin/env python3
"""
Get Telegram user info by user ID or username.
Uses existing Telegram credentials from .env file.

Usage:
    python get_username.py 123456789
    python get_username.py @username
    python get_username.py 123456789 --session custom_session
"""

import os
import re
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Union

from telethon import TelegramClient
from telethon.tl.types import User, PeerUser
from telethon.tl.functions.users import GetUsersRequest
from telethon.errors import UserIdInvalidError, UsernameNotOccupiedError, FloodWaitError
from dotenv import load_dotenv


def normalize_user_input(user_input: str) -> Union[int, str]:
    """Normalize user input to either user ID (int) or username (str)."""
    user_input = user_input.strip()
    
    # Check if it's a user ID (numeric)
    if user_input.isdigit():
        return int(user_input)
    
    # Check if it's a username format
    if user_input.startswith('@'):
        return user_input
    
    # If it looks like a username without @, add it
    if re.match(r'^[a-zA-Z0-9_]{5,32}$', user_input):
        return f'@{user_input}'
    
    # Try to parse as int if possible
    try:
        return int(user_input)
    except ValueError:
        # If all else fails, treat as username
        return user_input if user_input.startswith('@') else f'@{user_input}'


async def get_user_info(client: TelegramClient, user_input: Union[int, str]) -> Optional[dict]:
    """Get user info by Telegram user ID or username."""
    try:
        # Try multiple approaches to resolve the user
        if isinstance(user_input, int):
            try:
                # Method 1: Direct API call using GetUsersRequest
                result = await client(GetUsersRequest([PeerUser(user_input)]))
                if result:
                    user = result[0]
                else:
                    raise UserIdInvalidError("User not found")
            except Exception:
                try:
                    # Method 2: Try get_input_entity first
                    peer = await client.get_input_entity(user_input)
                    user = await client.get_entity(peer)
                except Exception:
                    # Method 3: Direct get_entity as last resort
                    user = await client.get_entity(user_input)
        else:
            # Handle special case for "me"
            if user_input in ['me', '@me']:
                user = await client.get_me()
            else:
                user = await client.get_entity(user_input)
        
        if isinstance(user, User):
            return {
                'user_id': user.id,
                'username': user.username,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'full_name': f"{user.first_name or ''} {user.last_name or ''}".strip() or None,
                'is_bot': user.bot,
                'is_verified': user.verified,
                'is_premium': getattr(user, 'premium', None),
                'lookup_input': user_input
            }
        else:
            return {
                'lookup_input': user_input,
                'error': f'Entity is not a user: {type(user).__name__}'
            }
            
    except UserIdInvalidError:
        return {
            'lookup_input': user_input,
            'error': 'User ID not found or invalid'
        }
    except UsernameNotOccupiedError:
        return {
            'lookup_input': user_input,
            'error': 'Username not found or not occupied'
        }
    except FloodWaitError as e:
        return {
            'lookup_input': user_input,
            'error': f'Rate limited. Wait {e.seconds} seconds and try again'
        }
    except Exception as e:
        return {
            'lookup_input': user_input,
            'error': str(e)
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Get Telegram user info by user ID or username",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 123456789                         # Get user info by user ID (user session)
  %(prog)s @username                         # Get user info by username (user session)
  %(prog)s 123456789 --bot-token 123:ABC... # Use bot token (user must have contacted bot)
  %(prog)s 123456789 --session custom       # Use custom session file
"""
    )
    
    parser.add_argument("user_input", help="Telegram user ID (number) or username (@username or username)")
    parser.add_argument(
        "--session",
        default="telegram_session", 
        help="Session file name (default: telegram_session)"
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("./export"),
        help="Directory containing session files (default: ./export)"
    )
    parser.add_argument(
        "--bot-token",
        help="Use bot token instead of user session (format: 123456:ABC-DEF...)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize client - either with bot token or user credentials
    if args.bot_token:
        # Use bot token
        api_id = os.getenv("TG_API_ID")
        api_hash = os.getenv("TG_API_HASH")
        
        if not api_id or not api_hash:
            print("Error: TG_API_ID and TG_API_HASH must be set in .env file even when using bot token")
            return 1
            
        try:
            api_id = int(api_id)
        except ValueError:
            print("Error: TG_API_ID must be a number")
            return 1
            
        client = TelegramClient('bot_session', api_id, api_hash)
    else:
        # Use user session
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
        
        # Create session directory if needed
        args.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        session_path = args.session_dir / args.session
        client = TelegramClient(str(session_path), api_id, api_hash)
    
    try:
        # Normalize user input
        user_lookup = normalize_user_input(args.user_input)
        
        if isinstance(user_lookup, int):
            print(f"Looking up user ID: {user_lookup}")
        else:
            print(f"Looking up username: {user_lookup}")
            
        if args.bot_token:
            await client.start(bot_token=args.bot_token)
        else:
            await client.start()
        
        result = await get_user_info(client, user_lookup)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return 1
        else:
            print("✓ User found:")
            print(f"  User ID: {result['user_id']}")
            print(f"  Username: @{result['username']}" if result['username'] else "  Username: (none)")
            print(f"  Name: {result['full_name']}" if result['full_name'] else "  Name: (none)")
            if result['first_name']:
                print(f"  First name: {result['first_name']}")
            if result['last_name']:
                print(f"  Last name: {result['last_name']}")
            print(f"  Is bot: {result['is_bot']}")
            print(f"  Is verified: {result['is_verified']}")
            if result['is_premium'] is not None:
                print(f"  Is premium: {result['is_premium']}")
                
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        await client.disconnect()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)