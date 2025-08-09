#!/usr/bin/env python3
"""
Serial file merger using OpenRouter API.
Merges multiple markdown files sequentially using LLM.

Usage Examples:
---------------

# Basic merge (minimal parameters)
python merge_files.py export/analysis_022.md export/analysis_010.md

# Full debug mode (all features for initial testing)
python merge_files.py export/analysis_022.md export/analysis_010.md \
    --save-intermediate \
    --save-responses \
    --interactive \
    --output-dir debug_output/

# Production run with auto-confirmation
python merge_files.py export/analysis_022.md export/analysis_010.md \
    --save-intermediate \
    --output-dir results/ \
    --yes

# Resume from failure (continue from last successful merge)
python merge_files.py export/analysis_015.md export/analysis_010.md \
    --base-file export/analysis_merged_step_005.md \
    --save-intermediate

# Custom prompt template
python merge_files.py report_10.txt report_01.txt \
    --prompt-file custom_merge_prompt.md \
    --save-intermediate

# All parameters example
python merge_files.py export/analysis_022.md export/analysis_010.md \
    --base-file export/base.md \
    --output-dir output/ \
    --prompt-file merge-prompt.md \
    --save-intermediate \
    --save-responses \
    --interactive \
    --yes

Parameters:
-----------
  start                 Start file (e.g., analysis_022.md)
  end                   End file (e.g., analysis_010.md)
  --base-file FILE      Explicit base file to start from (for resuming)
  --output-dir DIR      Output directory (default: current directory)
  --prompt-file FILE    Path to prompt template (default: merge-prompt.md)
  --save-intermediate   Save intermediate results after each merge
  --save-responses      Save raw API responses for debugging
  --interactive         Pause after each merge for user confirmation
  -y, --yes            Skip initial file list confirmation
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional
import requests
from dotenv import load_dotenv


class FileMerger:
    def __init__(self, args):
        """Initialize the merger with configuration."""
        self.args = args
        self.load_config()
        self.files_to_merge = []
        self.current_step = 0
        
    def load_config(self):
        """Load configuration from .env file."""
        load_dotenv()
        
        # API Configuration
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
        """Discover files to merge based on start and end parameters."""
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
        
        # Extract number from filename (assumes pattern like name_NNN)
        start_match = re.search(r'_(\d+)$', start_name)
        end_match = re.search(r'_(\d+)$', end_name)
        
        if not start_match or not end_match:
            print("❌ Error: Could not extract numbers from filenames")
            print("  Expected pattern: name_NNN.ext")
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
        print("📁 Files to merge (in order):")
        print("-" * 50)
        
        for i, file_path in enumerate(files, 1):
            size = file_path.stat().st_size
            size_kb = size / 1024
            print(f"  {i:2d}. {file_path.name:30s} ({size_kb:8.1f} KB)")
            
        print("-" * 50)
        print(f"  Total files: {len(files)}")
        print()
        
        if self.args.yes:
            return True
            
        response = input("Proceed with merging? (y/n): ").strip().lower()
        return response == 'y'
        
    def load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        prompt_file = Path(self.args.prompt_file)
        
        if not prompt_file.exists():
            print(f"❌ Error: Prompt file not found: {prompt_file}")
            sys.exit(1)
            
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()
            
        # Check for required placeholders
        if '{base_file_content}' not in template or '{merge_file_content}' not in template:
            print("❌ Error: Prompt template must contain {base_file_content} and {merge_file_content} placeholders")
            sys.exit(1)
            
        return template
        
    def call_openrouter_api(self, prompt: str) -> Optional[str]:
        """Call OpenRouter API with the merge prompt."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/user/telegram-merger',
            'X-Title': 'Telegram Analysis Merger'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert content merger. Output only the merged content without any explanations or meta-comments.'
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
            try:
                print(f"  📡 Calling API (attempt {attempt + 1}/{self.max_retries})...")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=300  # 5 minute timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Save raw response if configured
                    if self.args.save_responses:
                        response_file = self.get_output_path(f"response_step_{self.current_step:03d}.json")
                        with open(response_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                            
                    # Extract content
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0]['message']['content']
                        return content
                    else:
                        print(f"  ⚠️  Unexpected API response structure")
                        
                else:
                    print(f"  ⚠️  API error: {response.status_code}")
                    print(f"     {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                print(f"  ⚠️  Request timeout")
            except Exception as e:
                print(f"  ⚠️  Error: {str(e)}")
                
            if attempt < self.max_retries - 1:
                print(f"  ⏳ Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)
                
        return None
        
    def validate_merge_result(self, base_content: str, merged_content: str) -> bool:
        """Validate that the merge result is reasonable."""
        base_size = len(base_content)
        merged_size = len(merged_content)
        
        if merged_size < base_size:
            print(f"  ❌ Validation failed: Output ({merged_size} chars) is smaller than input ({base_size} chars)")
            return False
            
        print(f"  ✓ Validation passed: {base_size} → {merged_size} chars (+{merged_size - base_size})")
        return True
        
    def get_output_path(self, filename: str) -> Path:
        """Get the output file path."""
        if self.args.output_dir:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / filename
        else:
            return Path(filename)
            
    def merge_two_files(self, base_file: Path, merge_file: Path) -> Optional[str]:
        """Merge two files using the API."""
        print(f"\n📝 Merging: {merge_file.name} → {base_file.name}")
        
        # Load file contents
        with open(base_file, 'r', encoding='utf-8') as f:
            base_content = f.read()
            
        with open(merge_file, 'r', encoding='utf-8') as f:
            merge_content = f.read()
            
        print(f"  Base file: {len(base_content)} chars")
        print(f"  Merge file: {len(merge_content)} chars")
        
        # Prepare prompt
        prompt = self.prompt_template.replace('{base_file_content}', base_content)
        prompt = prompt.replace('{merge_file_content}', merge_content)
        
        # Call API
        result = self.call_openrouter_api(prompt)
        
        if result is None:
            print("  ❌ API call failed after all retries")
            return None
            
        # Validate result
        if not self.validate_merge_result(base_content, result):
            return None
            
        return result
        
    def run(self):
        """Main execution flow."""
        print("🚀 File Merger Starting")
        print("=" * 60)
        
        # Discover files
        if self.args.base_file:
            print(f"📌 Using explicit base file: {self.args.base_file}")
            base_file = Path(self.args.base_file)
            if not base_file.exists():
                print(f"❌ Error: Base file not found: {base_file}")
                sys.exit(1)
            files = self.discover_files()
            # Remove base file from list if it's there
            files = [f for f in files if f != base_file]
        else:
            files = self.discover_files()
            if len(files) < 2:
                print("❌ Error: Need at least 2 files to merge")
                sys.exit(1)
            base_file = files[0]
            files = files[1:]
            
        # Show files and get confirmation
        print(f"\n📄 Base file: {base_file.name} ({base_file.stat().st_size / 1024:.1f} KB)")
        print()
        
        if not self.show_files_for_confirmation(files):
            print("❌ Merge cancelled by user")
            sys.exit(0)
            
        # Load prompt template
        self.prompt_template = self.load_prompt_template()
        print(f"✓ Prompt template loaded from: {self.args.prompt_file}")
        
        # Extract base name for output files
        base_name = base_file.stem
        if '_' in base_name:
            base_name = base_name.rsplit('_', 1)[0]
            
        # Prepare base content
        current_content = base_file.read_text(encoding='utf-8')
        
        # Process each file
        total_files = len(files)
        
        for i, merge_file in enumerate(files, 1):
            self.current_step = i
            
            print(f"\n{'=' * 60}")
            print(f"Step {i}/{total_files}: Processing {merge_file.name}")
            print(f"{'=' * 60}")
            
            # Merge
            merged_content = self.merge_two_files(
                base_file if i == 1 else Path('temp_base.md'),
                merge_file
            )
            
            if merged_content is None:
                print(f"\n❌ Merge failed at step {i}")
                print(f"💡 To continue from here:")
                print(f"   Use --base-file with the last successful output")
                print(f"   Start with: {merge_file.name}")
                sys.exit(1)
                
            # Save result
            if self.args.save_intermediate or i == total_files:
                if self.args.save_intermediate:
                    output_file = self.get_output_path(f"{base_name}_merged_step_{i:03d}.md")
                else:
                    output_file = self.get_output_path(f"{base_name}_merged_final.md")
                    
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(merged_content)
                    
                print(f"  💾 Saved: {output_file}")
                
            # Update current content for next iteration
            current_content = merged_content
            
            # Save temp file for next iteration
            if i < total_files:
                with open('temp_base.md', 'w', encoding='utf-8') as f:
                    f.write(current_content)
                    
            # Interactive mode
            if self.args.interactive and i < total_files:
                print(f"\n⏸️  Step {i} completed. Press Enter to continue or Ctrl+C to stop...")
                input()
                
        # Cleanup temp file
        if Path('temp_base.md').exists():
            Path('temp_base.md').unlink()
            
        print(f"\n{'=' * 60}")
        print(f"✅ Merge completed successfully!")
        print(f"   Total steps: {total_files}")
        
        if not self.args.save_intermediate:
            output_file = self.get_output_path(f"{base_name}_merged_final.md")
            print(f"   Final output: {output_file}")
        else:
            print(f"   All intermediate files saved")
            

def main():
    parser = argparse.ArgumentParser(
        description='Serial file merger using OpenRouter API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analysis_022.md analysis_010.md
  %(prog)s analysis_022.md analysis_010.md --save-intermediate
  %(prog)s analysis_015.md analysis_010.md --base-file merged_step_005.md
  %(prog)s report_10.txt report_01.txt --interactive
"""
    )
    
    # Required arguments
    parser.add_argument('start', help='Start file (e.g., analysis_022.md)')
    parser.add_argument('end', help='End file (e.g., analysis_010.md)')
    
    # Optional arguments
    parser.add_argument('--base-file', help='Explicit base file to start from (for resuming)')
    parser.add_argument('--output-dir', help='Output directory (default: current directory)')
    parser.add_argument('--prompt-file', default='merge-prompt.md', 
                        help='Path to prompt template file (default: merge-prompt.md)')
    parser.add_argument('--save-intermediate', action='store_true',
                        help='Save intermediate results after each merge')
    parser.add_argument('--save-responses', action='store_true',
                        help='Save raw API responses for debugging')
    parser.add_argument('--interactive', action='store_true',
                        help='Pause after each merge for user confirmation')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    # Run merger
    merger = FileMerger(args)
    
    try:
        merger.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Merge interrupted by user")
        print("💡 To continue from here, check the last saved file and use --base-file")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()