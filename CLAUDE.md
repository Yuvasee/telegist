# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Telegram channel/chat parser with LLM-powered discussion extraction. Parses messages from Telegram channels and creates narrative digests of discussions using a two-tier LLM approach.

## Project Structure

```
telegram-channel-parser/
├── src/                        # All source code
│   ├── telegram_parser.py      # Main parser CLI
│   ├── extraction_pipeline.py  # Two-tier LLM extraction
│   ├── semantic_chunker.py     # Message chunking
│   ├── preprocessor.py         # Deduplication
│   ├── cost_tracker.py         # API cost tracking
│   ├── find_chat.py            # Find chats by name
│   ├── send_to_telegram.py     # Send results to TG
│   └── providers/              # LLM API providers
│       ├── base.py             # Base provider interface
│       ├── config.py           # Model configuration
│       ├── openrouter.py       # OpenRouter provider
│       ├── gemini.py           # Google Gemini provider
│       └── anthropic.py        # Anthropic provider
├── legacy/                     # Old scripts (deprecated)
├── .env                        # API keys (not in git)
├── .env.example                # Environment template
└── requirements.txt            # Python dependencies
```

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

# Run extraction
python extraction_pipeline.py ../output/messages.jsonl -o ../output/extraction --show-cost

# Send result to Telegram
python send_to_telegram.py ../output/extraction/synthesis.md ../output
```

## Key Commands

All commands run from `src/` directory:

### Parsing Telegram Channels

```bash
# Parse public channel by username
python telegram_parser.py @channel_name --days 7 --output ../output

# Parse private channel by numeric ID
python telegram_parser.py -- -1001784493554 --days 7 --output ../output

# Find private channel ID by name
python find_chat.py "channel name" ../output
```

### Extraction Pipeline

```bash
# Full extraction with cost tracking
python extraction_pipeline.py ../output/messages.jsonl -o ../output/extraction --show-cost

# Custom models (defaults: gemini-2.0-flash-001 for T1, claude-sonnet-4 for T2)
python extraction_pipeline.py messages.jsonl --tier1-model openai/gpt-4o-mini --tier2-model anthropic/claude-sonnet-4.5

# Control parallelism
python extraction_pipeline.py messages.jsonl --max-parallel 3
```

### Sending Results

```bash
# Send synthesis to Telegram Saved Messages
python send_to_telegram.py ../output/extraction/synthesis.md ../output
```

## Environment Variables

Required in `.env`:

```bash
# Telegram API (from https://my.telegram.org)
TG_API_ID=...
TG_API_HASH=...

# OpenRouter API (for LLM access)
OPENROUTER_API_KEY=...

# Optional: Override default models
TIER1_MODEL=google/gemini-2.0-flash-001
TIER2_MODEL=anthropic/claude-sonnet-4
```

## Extraction Output Format

The pipeline produces **discussion-based narrative digests**:

```markdown
# Дайджест обсуждений

## Настройка Go в редакторе Zed

Участники обсуждали сложности работы с Go в Zed. @username сказал:
"короче go в zed это еще тот адище". В ответ поделились
[конфигурацией](https://github.com/...) для решения проблемы...

---

## Выбор мониторов

Развернулась дискуссия о curved мониторах. Пользователь 1000R поделился:
"Проблем из-за изгиба не замечал ни в играх ни в работе"...

---

**Период**: Jan 15-17, 2026
**Сообщений**: 200
```

**Language handling**: Russian chats → Russian output, others → English.

## Architecture

### Two-Tier Extraction

1. **Tier 1** (Gemini 2.0 Flash): Parallel extraction of discussion threads from chunks
   - Extracts: topic, participants, summary, quotes, links
   - ~$0.003 per 100 messages

2. **Tier 2** (Claude Sonnet 4): Synthesis into narrative digest
   - Merges related discussions
   - Creates readable narratives with embedded quotes
   - ~$0.06 per synthesis

### Data Flow

```
Telegram → Parser → JSONL → Preprocessor → Chunker → Tier1 → Tier2 → Digest
                                              ↓
                                    307 parallel extractions
```

## Output Structure

```
output/
├── messages.csv                    # Raw messages
├── messages.jsonl                  # For resume/processing
├── telegram_session.session        # Auth session (reusable)
└── extraction/
    ├── extraction_result.json      # Full structured data
    ├── synthesis.md                # Final digest
    └── cost_summary.txt            # Token/cost breakdown
```

## Cost Reference

| Dataset Size | Chunks | Cost | Duration |
|-------------|--------|------|----------|
| 100 messages | ~14 | $0.07 | 1m |
| 200 messages | ~35 | $0.15 | 2m |
| 2000 messages | ~300 | $0.90 | 4m |

## Important Notes

- **Session files**: Reuse `telegram_session.session` across outputs to avoid re-auth
- **Rate limits**: Parser handles FloodWaitError gracefully
- **Resume**: Parser resumes from last message ID if interrupted
- **Numeric IDs**: Use `--` before negative IDs: `python telegram_parser.py -- -1001234567`
- **Output folders**: All `*_output/` patterns are gitignored

## Development

When modifying code:
1. All source in `src/` - imports are relative within that directory
2. Test with small dataset first (`--days 1`)
3. Check costs with `--show-cost` flag
4. Legacy scripts in `legacy/` are deprecated, don't use
