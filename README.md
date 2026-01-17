# Telegist

Telegram channel/chat parser with LLM-powered discussion extraction. Parses messages from Telegram channels and creates narrative digests of discussions using a two-tier LLM approach.

## Features

- Export channel messages to CSV and JSONL formats
- **Two-tier LLM extraction**: Fast chunk processing + quality synthesis
- **Discussion-based narratives**: Stories with participants, quotes, and links
- **Language-aware**: Russian chats → Russian output
- Resume interrupted exports
- Cost tracking for API usage
- Support for private channels (by numeric ID)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup credentials
cp .env.example .env
# Edit .env: add TG_API_ID, TG_API_HASH, OPENROUTER_API_KEY

# Run from src/ directory
cd src

# Parse channel (last 2 days)
python telegram_parser.py @channel_name --days 2 --output ../output

# Run extraction pipeline
python extraction_pipeline.py ../output/messages.jsonl -o ../output/extraction --show-cost

# Send result to Telegram Saved Messages
python send_to_telegram.py ../output/extraction/synthesis.md ../output
```

## Project Structure

```
telegist/
├── src/                        # All source code
│   ├── telegram_parser.py      # Main parser CLI
│   ├── extraction_pipeline.py  # Two-tier LLM extraction
│   ├── semantic_chunker.py     # Message chunking
│   ├── preprocessor.py         # Deduplication
│   ├── cost_tracker.py         # API cost tracking
│   ├── find_chat.py            # Find chats by name
│   ├── send_to_telegram.py     # Send results to TG
│   └── providers/              # LLM API providers
├── output/                     # Parser output (gitignored)
├── legacy/                     # Old scripts (deprecated)
├── .env                        # API keys (not in git)
└── requirements.txt            # Python dependencies
```

## Environment Variables

Required in `.env`:

```bash
# Telegram API (from https://my.telegram.org)
TG_API_ID=...
TG_API_HASH=...

# OpenRouter API (for LLM access)
OPENROUTER_API_KEY=...
```

## Commands

All commands run from `src/` directory:

### Parsing

```bash
# Parse public channel
python telegram_parser.py @channel_name --days 7 --output ../output

# Parse private channel by numeric ID
python telegram_parser.py -- -1001784493554 --days 7 --output ../output

# Find private channel ID by name
python find_chat.py "channel name" ../output
```

### Extraction

```bash
# Full extraction with cost tracking
python extraction_pipeline.py ../output/messages.jsonl -o ../output/extraction --show-cost
```

### Send Results

```bash
# Send synthesis to Telegram Saved Messages
python send_to_telegram.py ../output/extraction/synthesis.md ../output
```

## Output Format

The pipeline produces **discussion-based narrative digests**:

```markdown
# Дайджест обсуждений

## Настройка Go в редакторе Zed

Участники обсуждали сложности работы с Go в Zed. @username сказал:
"короче go в zed это еще тот адище". В ответ поделились
[конфигурацией](https://github.com/...) для решения проблемы...

---

**Период**: Jan 15-17, 2026
**Сообщений**: 200
```

## Two-Tier Architecture

1. **Tier 1** (Gemini 2.0 Flash): Parallel extraction from chunks
   - Extracts: topic, participants, summary, quotes, links
   - ~$0.003 per 100 messages

2. **Tier 2** (Claude Sonnet 4): Synthesis into narrative
   - Merges related discussions
   - Creates readable narratives
   - ~$0.06 per synthesis

## Cost Reference

| Dataset Size | Chunks | Cost | Duration |
|-------------|--------|------|----------|
| 100 messages | ~14 | $0.07 | 1m |
| 200 messages | ~35 | $0.15 | 2m |
| 2000 messages | ~300 | $0.90 | 4m |

## Notes

- **Session files**: Reuse `telegram_session.session` across runs to avoid re-auth
- **Rate limits**: Parser handles FloodWaitError gracefully
- **Resume**: Parser resumes from last message ID if interrupted
- **Numeric IDs**: Use `--` before negative IDs: `python telegram_parser.py -- -1001234567`

## License

MIT
