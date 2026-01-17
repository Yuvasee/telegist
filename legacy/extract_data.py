#!/usr/bin/env python3
"""
Telegram Messages Data Extractor using OpenRouter API.
Processes CSV files from telegram_parser.py and extracts structured insights using LLM.

Usage Examples:
---------------

# Basic extraction (process all files from 001 to 043)
python extract_data.py messages_001.csv messages_043.csv

# Process specific range with custom output directory
python extract_data.py export/messages_001.csv export/messages_010.csv \
    --output-dir analysis/

# Full debug mode (all features for testing)
python extract_data.py messages_001.csv messages_005.csv \
    --save-responses \
    --interactive \
    --output-dir debug_output/

# Production run with auto-confirmation (default 3 parallel workers)
python extract_data.py messages_001.csv messages_043.csv \
    --output-dir results/ \
    --yes

# Custom prompt template with high parallelism
python extract_data.py messages_001.csv messages_010.csv \
    --prompt-file custom_extract_prompt.md \
    --save-responses \
    --max-parallel 5

# Skip already processed files (sequential for safety)
python extract_data.py messages_001.csv messages_043.csv \
    --skip-existing \
    --output-dir analysis/ \
    --max-parallel 1

# All parameters example
python extract_data.py export/messages_001.csv export/messages_043.csv \
    --output-dir output/ \
    --prompt-file extract-prompt.md \
    --save-responses \
    --interactive \
    --skip-existing \
    --yes

Parameters:
-----------
  start                 Start CSV file (e.g., messages_001.csv)
  end                   End CSV file (e.g., messages_043.csv)
  --output-dir DIR      Output directory for analysis files (default: current directory)
  --prompt-file FILE    Path to prompt template (default: extract-prompt.md)
  --save-responses      Save raw API responses for debugging
  --interactive         Pause after each extraction for user confirmation (forces sequential processing)
  --skip-existing       Skip files that already have analysis output
  -y, --yes            Skip initial file list confirmation
  --max-parallel        Maximum parallel file processing workers (default: 3)

Notes:
------
- Processes only 'text' column from CSV files
- Creates analysis_NNN.md files from messages_NNN.csv files
- Uses same OpenRouter API configuration as merge_files.py
- Processes files in parallel by default (3 workers) for faster processing
- Retries 3 times per file on API failure, continues with other files
- Collects and reports all failures at the end
- Interactive mode forces sequential processing (1 worker)
- Overwrites existing analysis files unless --skip-existing is used
"""

import os
import sys
import json
import csv
import time
import argparse
import re
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Suppress the urllib3 OpenSSL warning
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import requests
from dotenv import load_dotenv


class DataExtractor:
    def __init__(self, args):
        """Initialize the extractor with configuration."""
        self.args = args
        self.load_config()
        self.files_to_process = []
        self.current_file_index = 0
        self.api_lock = threading.Lock()  # For rate limiting
        self.progress_lock = threading.Lock()  # For progress reporting
        
    def load_config(self):
        """Load configuration from .env file."""
        load_dotenv()
        
        # API Configuration (same as merge_files.py)
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            print("❌ Error: OPENROUTER_API_KEY not found in .env file")
            sys.exit(1)
            
        self.api_url = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')
        self.model = os.getenv('OPENROUTER_MODEL', 'qwen/qwen-2.5-72b-instruct')
        self.temperature = float(os.getenv('OPENROUTER_TEMPERATURE', '0.3'))
        self.max_retries = int(os.getenv('OPENROUTER_MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('OPENROUTER_RETRY_DELAY', '5'))
        
        # Max tokens - None means no limit
        max_tokens_str = os.getenv('OPENROUTER_MAX_TOKENS', 'none')
        self.max_tokens = None if max_tokens_str.lower() == 'none' else int(max_tokens_str)
        
        print(f"✓ Configuration loaded")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {'unlimited' if self.max_tokens is None else self.max_tokens}")
        print()
        
    def discover_files(self) -> List[Path]:
        """Discover CSV files to process based on start and end parameters."""
        start_path = Path(self.args.start)
        end_path = Path(self.args.end)
        
        if not start_path.exists():
            print(f"❌ Error: Start file not found: {start_path}")
            sys.exit(1)
            
        if not end_path.exists():
            print(f"❌ Error: End file not found: {end_path}")
            sys.exit(1)
            
        # Extract pattern from filenames
        start_name = start_path.stem
        end_name = end_path.stem
        
        # Extract number from filename (assumes pattern like messages_NNN)
        start_match = re.search(r'_(\d+)$', start_name)
        end_match = re.search(r'_(\d+)$', end_name)
        
        if not start_match or not end_match:
            print("❌ Error: Could not extract numbers from filenames")
            print("  Expected pattern: messages_NNN.csv")
            sys.exit(1)
            
        start_num = int(start_match.group(1))
        end_num = int(end_match.group(1))
        
        # Extract base name (everything before _NNN)
        base_name = start_name[:start_match.start()]
        extension = start_path.suffix
        directory = start_path.parent
        
        # Determine order
        if start_num > end_num:
            step = -1
            nums = range(start_num, end_num - 1, step)
        else:
            step = 1
            nums = range(start_num, end_num + 1, step)
            
        # Collect existing files
        files = []
        for num in nums:
            file_path = directory / f"{base_name}_{num:03d}{extension}"
            if file_path.exists():
                files.append(file_path)
            else:
                print(f"  ⚠️  Skipping missing file: {file_path.name}")
                
        return files
        
    def show_files_for_confirmation(self, files: List[Path]) -> bool:
        """Display discovered files and ask for confirmation."""
        print("📁 CSV files to process (in order):")
        print("-" * 50)
        
        total_messages = 0
        for i, file_path in enumerate(files, 1):
            size = file_path.stat().st_size
            size_kb = size / 1024
            
            # Count messages in file
            message_count = self.count_messages_in_csv(file_path)
            total_messages += message_count
            
            # Check if output exists
            output_file = self.get_output_filename(file_path)
            exists_marker = " [✓ exists]" if output_file.exists() else ""
            
            print(f"  {i:2d}. {file_path.name:30s} ({size_kb:8.1f} KB, {message_count:4d} messages){exists_marker}")
            
        print("-" * 50)
        print(f"  Total files: {len(files)}")
        print(f"  Total messages: {total_messages}")
        
        if self.args.skip_existing:
            existing_count = sum(1 for f in files if self.get_output_filename(f).exists())
            print(f"  Will skip {existing_count} existing analysis files")
        
        print()
        
        if self.args.yes:
            return True
            
        response = input("Proceed with extraction? (y/n): ").strip().lower()
        return response == 'y'
        
    def count_messages_in_csv(self, file_path: Path) -> int:
        """Count non-empty messages in a CSV file."""
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('text', '').strip():
                        count += 1
        except Exception:
            pass
        return count
        
    def load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        prompt_file = Path(self.args.prompt_file)
        
        if not prompt_file.exists():
            print(f"❌ Error: Prompt file not found: {prompt_file}")
            print("  Creating default prompt template...")
            self.create_default_prompt_template(prompt_file)
            print(f"  ✓ Created {prompt_file}")
            print("  Please review and adjust the prompt if needed, then run again.")
            sys.exit(0)
            
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()
            
        # Check for required placeholder
        if '{messages_content}' not in template:
            print("❌ Error: Prompt template must contain {messages_content} placeholder")
            sys.exit(1)
            
        return template
        
    def create_default_prompt_template(self, prompt_file: Path):
        """Create a default extraction prompt template."""
        default_prompt = """# Инструкции по анализу сообщений

Ты эксперт-аналитик, специализирующийся на анализе контента из Telegram-каналов adult-индустрии.

## Сообщения для анализа:
{messages_content}

## Твоя задача:
Проанализируй предоставленные сообщения и извлеки из них ценную информацию, организовав её в структурированном виде.

## Что нужно извлечь:
1. **Практические советы и лайфхаки** - конкретные рекомендации по работе на платформах, продвижению, монетизации
2. **Технические детали** - настройки, инструменты, сервисы, платформы с конкретными названиями и параметрами
3. **Финансовая информация** - цены, доходы, расценки, проценты комиссий
4. **Истории и кейсы** - реальные примеры успехов и неудач с деталями
5. **Тренды и изменения** - новые функции платформ, изменения в алгоритмах, новые возможности
6. **Проблемы и решения** - типичные сложности и способы их преодоления
7. **Инструменты и сервисы** - конкретные названия, ссылки, описания функций

## Требования к результату:
- Структурируй информацию по тематическим разделам с заголовками
- Сохраняй ВСЕ конкретные детали: названия, числа, проценты, даты, суммы
- Используй списки и подзаголовки для удобной навигации
- Пиши на русском языке
- Формат: Markdown
- НЕ обобщай и не упрощай - сохраняй максимум конкретики
- Если в сообщениях есть ссылки, сохраняй их

## Важно:
- Выводи ТОЛЬКО структурированный анализ
- НЕ добавляй вступления типа "Вот анализ..." или заключения
- НЕ добавляй мета-комментарии о процессе анализа
- Начинай сразу с заголовка и контента анализа"""
        
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(default_prompt)
            
    def extract_messages_from_csv(self, csv_path: Path) -> str:
        """Extract text messages from CSV file."""
        messages = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get('text', '').strip()
                    if text:
                        # Replace escaped newlines with actual newlines
                        text = text.replace('\\n', '\n')
                        messages.append(text)
                        
        except Exception as e:
            print(f"  ❌ Error reading CSV file: {str(e)}")
            return ""
            
        # Join messages with clear separators
        return "\n\n---\n\n".join(messages)
        
    def call_openrouter_api(self, prompt: str, file_index: int, filename: str = "") -> Optional[str]:
        """Call OpenRouter API with the extraction prompt."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/user/telegram-extractor',
            'X-Title': 'Telegram Data Extractor'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'Ты эксперт-аналитик, извлекающий структурированную информацию из сообщений. Выводи только результат анализа без мета-комментариев.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': self.temperature
        }
        
        if self.max_tokens is not None:
            payload['max_tokens'] = self.max_tokens
            
        for attempt in range(self.max_retries):
            response = None
            try:
                with self.progress_lock:
                    print(f"  📡 {filename}: Calling API (attempt {attempt + 1}/{self.max_retries})...")
                
                # Rate limiting with lock
                with self.api_lock:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=300  # 5 minute timeout
                    )
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Save raw response if configured
                        if self.args.save_responses:
                            response_file = self.get_output_path(f"response_file_{file_index:03d}_attempt_{attempt + 1}.json")
                            with open(response_file, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                                
                        # Extract content
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['message']['content']
                            return content
                        else:
                            print(f"  ⚠️  Unexpected API response structure")
                            
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️  Invalid JSON response: {str(e)}")
                        
                else:
                    print(f"  ⚠️  API error: {response.status_code}")
                    print(f"     Response: {response.text[:200]}...")
                    
            except requests.exceptions.Timeout:
                print(f"  ⚠️  Request timeout")
            except Exception as e:
                print(f"  ⚠️  Error: {str(e)}")
                
            if attempt < self.max_retries - 1:
                print(f"  ⏳ Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)
                
        return None
        
    def get_output_path(self, filename: str) -> Path:
        """Get the output file path."""
        if self.args.output_dir:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / filename
        else:
            return Path(filename)
            
    def get_output_filename(self, csv_path: Path) -> Path:
        """Generate output filename from CSV filename."""
        # Extract number from messages_NNN.csv
        csv_name = csv_path.stem
        match = re.search(r'_(\d+)$', csv_name)
        
        if match:
            number = match.group(1)
            output_name = f"analysis_{number}.md"
        else:
            # Fallback if pattern doesn't match
            output_name = f"analysis_{csv_name}.md"
            
        return self.get_output_path(output_name)
        
    def process_csv_file_task(self, csv_path: Path, file_index: int, total_files: int) -> Tuple[bool, str, Optional[str]]:
        """Task wrapper for parallel processing. Returns (success, filename, error_message)."""
        try:
            success = self.process_csv_file(csv_path, file_index, total_files)
            return (success, csv_path.name, None)
        except Exception as e:
            return (False, csv_path.name, str(e))

    def process_csv_file(self, csv_path: Path, file_index: int, total_files: int = 1) -> bool:
        """Process a single CSV file and generate analysis."""
        output_file = self.get_output_filename(csv_path)
        
        # Check if should skip
        if self.args.skip_existing and output_file.exists():
            with self.progress_lock:
                print(f"  ⏭️  {csv_path.name}: Skipping (already exists)")
            return True
            
        with self.progress_lock:
            print(f"📝 [{file_index}/{total_files}] Processing: {csv_path.name}")
        
        # Extract messages
        messages_content = self.extract_messages_from_csv(csv_path)
        
        if not messages_content:
            with self.progress_lock:
                print(f"  ⚠️  {csv_path.name}: No messages found")
            return True  # Not a failure, just empty
            
        message_count = len(messages_content.split('\n\n---\n\n'))
        with self.progress_lock:
            print(f"  📊 {csv_path.name}: {message_count} messages, {len(messages_content)} characters")
        
        # Prepare prompt
        prompt = self.prompt_template.replace('{messages_content}', messages_content)
        
        # Call API
        result = self.call_openrouter_api(prompt, file_index, csv_path.name)
        
        if result is None:
            with self.progress_lock:
                print(f"  ❌ {csv_path.name}: API call failed after all retries")
            return False
            
        # Save result
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
            
        with self.progress_lock:
            print(f"  ✅ {csv_path.name}: Saved {output_file.name} ({len(result)} characters)")
        
        return True
    
    def process_files_parallel(self, files: List[Path]) -> Tuple[int, int, int, List[str]]:
        """Process multiple files in parallel. Returns (successful, failed, skipped, error_messages)."""
        successful = 0
        failed = 0
        skipped = 0
        error_messages = []
        total_files = len(files)
        
        # Handle interactive mode - force sequential processing
        max_workers = 1 if self.args.interactive else self.args.max_parallel
        
        print(f"\n🔄 Processing {total_files} files with {max_workers} parallel worker(s)")
        print("-" * 60)
        
        # Pre-filter files that should be skipped
        files_to_process = []
        for i, csv_file in enumerate(files, 1):
            output_file = self.get_output_filename(csv_file)
            if self.args.skip_existing and output_file.exists():
                print(f"⏭️  [{i}/{total_files}] Skipping existing: {output_file.name}")
                skipped += 1
            else:
                files_to_process.append((csv_file, i))
        
        if not files_to_process:
            return successful, failed, skipped, error_messages
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for csv_file, file_index in files_to_process:
                future = executor.submit(self.process_csv_file_task, csv_file, file_index, total_files)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                success, filename, error_msg = future.result()
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    error_msg_formatted = f"{filename}: {error_msg}" if error_msg else f"{filename}: Unknown error"
                    error_messages.append(error_msg_formatted)
                    
                # Interactive mode handling
                if self.args.interactive and len(files_to_process) > 1:
                    with self.progress_lock:
                        remaining = len([f for f in futures if not f.done()])
                        if remaining > 0:
                            print(f"\n⏸️  File completed. {remaining} remaining. Press Enter to continue or Ctrl+C to stop...")
                            input()
        
        return successful, failed, skipped, error_messages
        
    def run(self):
        """Main execution flow."""
        print("🚀 Data Extractor Starting")
        print("=" * 60)
        
        # Discover files
        files = self.discover_files()
        
        if not files:
            print("❌ Error: No files found to process")
            sys.exit(1)
            
        # Show files and get confirmation
        if not self.show_files_for_confirmation(files):
            print("❌ Extraction cancelled by user")
            sys.exit(0)
            
        # Load prompt template
        self.prompt_template = self.load_prompt_template()
        print(f"✓ Prompt template loaded from: {self.args.prompt_file}")
        
        # Process files in parallel
        successful, failed, skipped, error_messages = self.process_files_parallel(files)
        total_files = len(files)
                
        # Summary
        print(f"\n{'=' * 60}")
        print(f"✅ Extraction completed!")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Skipped: {skipped}")
        print(f"   Total processed: {successful + failed + skipped}/{total_files}")
        
        if error_messages:
            print(f"\n❌ Failed files:")
            for error_msg in error_messages:
                print(f"   • {error_msg}")
        
        if failed > 0:
            print(f"\n⚠️  {failed} files failed to process.")
            print(f"   You can retry with the same command to process failed files.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Extract structured data from Telegram message CSV files using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s messages_001.csv messages_043.csv
  %(prog)s messages_001.csv messages_010.csv --save-responses
  %(prog)s export/messages_001.csv export/messages_043.csv --output-dir analysis/
  %(prog)s messages_001.csv messages_005.csv --interactive --skip-existing
"""
    )
    
    # Required arguments
    parser.add_argument('start', help='Start CSV file (e.g., messages_001.csv)')
    parser.add_argument('end', help='End CSV file (e.g., messages_043.csv)')
    
    # Optional arguments
    parser.add_argument('--output-dir', help='Output directory for analysis files (default: current directory)')
    parser.add_argument('--prompt-file', default='extract-prompt.md', 
                        help='Path to prompt template file (default: extract-prompt.md)')
    parser.add_argument('--save-responses', action='store_true',
                        help='Save raw API responses for debugging')
    parser.add_argument('--interactive', action='store_true',
                        help='Pause after each extraction for user confirmation')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that already have analysis output')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--max-parallel', type=int, default=3,
                        help='Maximum parallel file processing (default: 3)')
    
    args = parser.parse_args()
    
    # Run extractor
    extractor = DataExtractor(args)
    
    try:
        extractor.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()