# Telegram Channel Parser

Export messages from public Telegram channels to CSV and JSONL formats. Supports resuming interrupted exports and optional media download.

## Features

- Export channel messages to CSV and JSONL formats
- Resume from last exported message if interrupted
- Optional media download (photos, documents)
- **Split large CSV files** into manageable chunks for LLM analysis
- Progress bar with real-time updates
- Handles rate limits gracefully
- CLI interface with channel as parameter
- Credentials stored securely in `.env` file

## Prerequisites

1. Python 3.7+
2. Telegram API credentials (API ID and API Hash)

## Setup

### 1. Get Telegram API Credentials

1. Go to https://my.telegram.org
2. Log in with your phone number
3. Navigate to "API development tools"
4. Create an app to get your `API_ID` and `API_HASH`

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your credentials
# TG_API_ID=123456
# TG_API_HASH=your_api_hash_here
```

## Usage

### Basic Export (Text Only)

```bash
python telegram_parser.py @channel_name
```

### Export with Media

```bash
python telegram_parser.py @channel_name --with-media
```

### Custom Output Directory

```bash
python telegram_parser.py @channel_name --output ./my_export
```

### Split Large CSV Files

For easier LLM analysis, you can split CSV files into smaller chunks:

```bash
# Split into files with 1000 messages each
python telegram_parser.py @channel_name --split-size 1000

# Split into files with 500 messages each
python telegram_parser.py @channel_name --split-size 500
```

### Examples

```bash
# Export BBC News channel
python telegram_parser.py @bbcnews

# Export with media to custom directory
python telegram_parser.py @bbcnews --with-media --output ./bbc_export

# Using t.me link format
python telegram_parser.py t.me/bbcnews

# Smaller batch size (useful for channels with rate limits)
python telegram_parser.py @channel_name --page-size 50

# Split large channel into 2000-message CSV files
python telegram_parser.py @large_channel --split-size 2000
```

## Output

The script creates the following files in the output directory:

- `messages.csv` - Spreadsheet-friendly format with one message per row (or `messages_001.csv`, `messages_002.csv`, etc. when using `--split-size`)
- `messages.jsonl` - JSON Lines format with full message data
- `media/` - Downloaded media files (if `--with-media` is used)
- `telegram_session.session` - Session file for resuming

### CSV Columns

- `id` - Message ID
- `date_iso` - Message timestamp (ISO format)
- `from_id` - Sender user ID
- `from_name` - Sender username or name
- `text` - Message text
- `views` - View count
- `forwards` - Forward count
- `replies` - Reply count
- `media_type` - Type of attached media
- `media_path` - Path to downloaded media file
- `link` - Direct link to message

## Resuming Exports

The script automatically resumes from the last exported message if interrupted. Just run the same command again:

```bash
# First run (interrupted with Ctrl+C)
python telegram_parser.py @large_channel

# Resume from where it left off
python telegram_parser.py @large_channel
```

## CSV Splitter Utility

For users who have already exported data and want to split existing CSV files, use the `csv_splitter.py` utility:

```bash
# Split existing CSV into 1000-message files
python csv_splitter.py messages.csv --split-size 1000

# Split with custom output directory and prefix
python csv_splitter.py messages.csv --split-size 500 --output-dir ./split --prefix export

# See all options
python csv_splitter.py --help
```

The splitter will create files like `messages_001.csv`, `messages_002.csv`, etc., each with the same CSV headers as the original file.

## Command-Line Options

```
positional arguments:
  channel               Channel to export (@username or t.me/slug)

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output directory (default: ./export)
  --with-media          Download media files (photos, documents)
  --page-size SIZE      Messages per batch (default: 100)
  --session NAME        Session file name (default: telegram_session)
  --split-size SIZE     Split CSV into files with max N messages each
```

## Notes

- **Rate Limits**: The script handles rate limits gracefully. If you hit a flood wait, just rerun the command
- **Large Channels**: Exporting channels with many messages can take hours, especially with media
- **Private Channels**: This script only works with public channels or channels you've already joined
- **Resuming**: The script tracks progress in the JSONL file and resumes from the last message ID

## Legal & Ethical Considerations

- Only export channels you have permission to archive
- Respect channel terms of service and local laws
- Don't use for spam, harassment, or unauthorized redistribution
- Be mindful of rate limits and Telegram's terms of service

## Troubleshooting

### "Set TG_API_ID and TG_API_HASH env vars first"
Make sure your `.env` file exists and contains valid credentials.

### FloodWaitError
You're hitting rate limits. Wait the specified time and rerun the command.

### Can't find channel
Ensure the channel is public and use the correct format: `@channel_name` or `t.me/channel_name`

### Session file issues
Delete the `.session` file in the output directory and try again.