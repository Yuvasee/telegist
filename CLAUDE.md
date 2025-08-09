# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Telegram channel parser that exports messages from public Telegram channels to CSV and JSONL formats. The project focuses on data extraction and analysis, particularly for adult content creator industry research.

## Key Commands

### Running the Parser
```bash
# Basic export
python telegram_parser.py @channel_name

# Export with media download
python telegram_parser.py @channel_name --with-media

# Export with CSV splitting for LLM analysis
python telegram_parser.py @channel_name --split-size 500

# Custom output directory
python telegram_parser.py @channel_name --output ./custom_dir
```

### Splitting Existing CSV Files
```bash
# Split existing CSV file
python csv_splitter.py export/messages.csv --split-size 500

# With custom output
python csv_splitter.py messages.csv --split-size 1000 --output-dir ./split --prefix export
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup credentials
cp .env.example .env
# Then edit .env to add TG_API_ID and TG_API_HASH from https://my.telegram.org
```

## Architecture & Key Components

### Main Scripts
- **telegram_parser.py**: Core parser with Telethon client that handles channel export, resuming, and optional CSV splitting
  - Uses `CSVWriter` class for intelligent file splitting
  - Tracks progress in JSONL for resuming interrupted exports
  - Handles rate limits and authentication including 2FA

- **csv_splitter.py**: Standalone utility for splitting existing CSV files
  - Preserves headers in each split file
  - Uses 3-digit zero-padded numbering for proper sorting

### Data Flow
1. **Authentication**: Uses Telegram API credentials from `.env` file
2. **Session Management**: Creates session files in output directory for resuming
3. **Message Fetching**: Paginated fetching with configurable batch size (default 100)
4. **Data Export**: Parallel writing to CSV and JSONL formats
5. **Resume Capability**: Uses last message ID from JSONL to continue from interruption

### Output Structure
```
export/
├── messages.csv (or messages_001.csv, messages_002.csv... when split)
├── messages.jsonl (single file, used for resume tracking)
├── media/ (optional, when --with-media is used)
└── telegram_session.session (authentication session)
```

## CSV Analysis Workflow

When analyzing exported data with LLMs:
1. Use `--split-size 500` for optimal chunk size (100-200KB per file)
2. Files are numbered sequentially (messages_001.csv, messages_002.csv, etc.)
3. Each file contains complete messages with all headers
4. Analysis results typically saved as analysis_XXX.md files

## Important Notes

- **Rate Limits**: Script handles FloodWaitError gracefully - just rerun if hit
- **2FA Password**: If prompted for password during first run, it's the Telegram account 2FA password (not API credentials)
- **Resume Feature**: Always resumes from last message ID - safe to interrupt with Ctrl+C
- **CSV Splitting**: JSONL remains single file for resume tracking; only CSV is split