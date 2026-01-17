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

# Tree-based merging (parallel processing, keep intermediate files)
python merge_files.py export/analysis_022.md export/analysis_010.md \
    --merge-strategy tree \
    --save-intermediate \
    --max-parallel 5

# Tree merging with state resuming (auto-cleanup intermediate files)
python merge_files.py export/analysis_022.md export/analysis_010.md \
    --merge-strategy tree \
    --state-file tree_merge_state.json

# All parameters example (sequential)
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
  --save-intermediate   Save intermediate results after each merge (tree mode: keep files after completion)
  --save-responses      Save raw API responses for debugging
  --interactive         Pause after each merge for user confirmation
  -y, --yes            Skip initial file list confirmation
  --merge-strategy      Choose merge algorithm: 'sequential' (default) or 'tree'
  --max-parallel        Maximum parallel API calls for tree merging (default: 3)
  --state-file          State file for resuming tree merges (auto-generated if not specified)
"""

import os
import sys
import json
import time
import argparse
import re
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading

# Suppress the urllib3 OpenSSL warning
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import requests
from dotenv import load_dotenv


class FileMerger:
    def __init__(self, args):
        """Initialize the merger with configuration."""
        self.args = args
        self.load_config()
        self.files_to_merge = []
        self.current_step = 0
        self.tree_state = None
        self.state_file = None
        self.api_lock = threading.Lock()  # For rate limiting
        self.state_lock = threading.Lock()  # For state file updates
        
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
            response = None
            try:
                print(f"  📡 Calling API (attempt {attempt + 1}/{self.max_retries})...")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=300  # 5 minute timeout
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Save raw response if configured
                        if self.args.save_responses:
                            response_file = self.get_output_path(f"response_step_{self.current_step:03d}_attempt_{attempt + 1}.json")
                            with open(response_file, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                                
                        # Extract content
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['message']['content']
                            return content
                        else:
                            print(f"  ⚠️  Unexpected API response structure")
                            if self.args.save_responses:
                                print(f"     Raw response saved for debugging")
                            
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️  Invalid JSON response: {str(e)}")
                        print(f"     Response preview: {response.text[:200]}...")
                        
                        # Save raw response for debugging
                        if self.args.save_responses:
                            response_file = self.get_output_path(f"response_step_{self.current_step:03d}_attempt_{attempt + 1}_raw.txt")
                            with open(response_file, 'w', encoding='utf-8') as f:
                                f.write(f"Status: {response.status_code}\n")
                                f.write(f"Headers: {dict(response.headers)}\n\n")
                                f.write(response.text)
                            print(f"     Full response saved to {response_file}")
                        
                else:
                    print(f"  ⚠️  API error: {response.status_code}")
                    print(f"     Response: {response.text[:200]}...")
                    
                    # Save error response for debugging
                    if self.args.save_responses:
                        response_file = self.get_output_path(f"response_step_{self.current_step:03d}_attempt_{attempt + 1}_error.txt")
                        with open(response_file, 'w', encoding='utf-8') as f:
                            f.write(f"Status: {response.status_code}\n")
                            f.write(f"Headers: {dict(response.headers)}\n\n")
                            f.write(response.text)
                    
            except requests.exceptions.Timeout:
                print(f"  ⚠️  Request timeout")
            except Exception as e:
                print(f"  ⚠️  Error: {str(e)}")
                # For unexpected errors, try to save what we can
                if self.args.save_responses and response is not None:
                    try:
                        response_file = self.get_output_path(f"response_step_{self.current_step:03d}_attempt_{attempt + 1}_exception.txt")
                        with open(response_file, 'w', encoding='utf-8') as f:
                            f.write(f"Exception: {str(e)}\n")
                            f.write(f"Response text: {response.text if hasattr(response, 'text') else 'No response'}\n")
                    except:
                        pass
                
            if attempt < self.max_retries - 1:
                print(f"  ⏳ Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)
                
        return None
        
    def validate_merge_result(self, base_content: str, merged_content: str, merge_content: str = None) -> bool:
        """Validate that the merge result is reasonable."""
        if self.args.merge_strategy == 'tree':
            # For tree merging, check against average of inputs
            total_input_size = len(base_content)
            if merge_content:  # 2-way merge
                total_input_size += len(merge_content)
                avg_size = total_input_size / 2
            else:
                # This case shouldn't happen in tree mode, but handle gracefully
                avg_size = len(base_content)
            
            merged_size = len(merged_content)
            min_acceptable_size = int(avg_size * 0.9)
            
            if merged_size < min_acceptable_size:
                print(f"  ❌ Validation failed: Output ({merged_size} chars) is much smaller than average input ({int(avg_size)} chars)")
                print(f"     Minimum acceptable: {min_acceptable_size} chars (90% of average)")
                return False
            
            print(f"  ✓ Validation passed: {int(avg_size)} avg → {merged_size} chars")
            return True
        else:
            # Original sequential validation
            base_size = len(base_content)
            merged_size = len(merged_content)
            
            # Allow some size reduction (up to 10%) in case of deduplication
            min_acceptable_size = int(base_size * 0.9)
            
            if merged_size < min_acceptable_size:
                print(f"  ❌ Validation failed: Output ({merged_size} chars) is much smaller than input ({base_size} chars)")
                print(f"     Minimum acceptable: {min_acceptable_size} chars (90% of base)")
                return False
                
            if merged_size < base_size:
                print(f"  ⚠️  Output slightly smaller: {base_size} → {merged_size} chars (-{base_size - merged_size})")
                print(f"     This might be due to deduplication, accepting...")
            else:
                print(f"  ✓ Validation passed: {base_size} → {merged_size} chars (+{merged_size - base_size})")
            
            return True
        
    def build_merge_tree(self, files: List[Path]) -> List[List[Tuple]]:
        """Build a tree structure for merging files.
        Returns a list of levels, where each level contains tuples of files to merge.
        """
        tree = []
        current_level = [(f,) for f in files]  # Start with individual files as tuples
        level_num = 0
        
        while len(current_level) > 1:
            next_level = []
            i = 0
            
            while i < len(current_level):
                remaining = len(current_level) - i
                
                if remaining >= 3 and remaining % 2 == 1:
                    # If odd number of items left and at least 3, merge three
                    next_level.append((current_level[i], current_level[i + 1], current_level[i + 2]))
                    i += 3
                elif remaining >= 2:
                    # If even number or exactly 2 left, merge pair
                    next_level.append((current_level[i], current_level[i + 1]))
                    i += 2
                else:
                    # Single item left - shouldn't happen with proper logic above
                    break
                    
            tree.append(current_level)
            current_level = next_level
            level_num += 1
            
        if current_level:
            tree.append(current_level)
            
        return tree
    
    def get_tree_output_filename(self, level: int, pair_num: int, files_tuple: Tuple) -> str:
        """Generate a readable filename for tree merge outputs."""
        base_name = Path(self.args.start).stem
        if '_' in base_name:
            base_name = base_name.rsplit('_', 1)[0]
        
        # For level 0 (original files), extract numbers
        if level == 0:
            nums = []
            for f in files_tuple:
                if isinstance(f, Path):
                    match = re.search(r'_(\d+)$', f.stem)
                    if match:
                        nums.append(match.group(1))
            if nums:
                return f"{base_name}_L{level}_{'-'.join(nums)}.md"
        
        # For higher levels, use level and pair number
        return f"{base_name}_L{level}_P{pair_num:02d}.md"
    
    def save_tree_state(self, state: Dict[str, Any]):
        """Save the current tree merging state to a file."""
        with self.state_lock:  # Thread-safe state saving
            if not self.state_file:
                if self.args.state_file:
                    self.state_file = Path(self.args.state_file)
                else:
                    base_name = Path(self.args.start).stem
                    if '_' in base_name:
                        base_name = base_name.rsplit('_', 1)[0]
                    self.state_file = self.get_output_path(f"{base_name}_tree_state.json")
            
            # Always use the new format
            unified_state = {
                'format_version': '2.0',
                'current_level': state.get('current_level', state.get('level', 0)),
                'completed_tasks': state.get('completed_tasks', {}),
                'timestamp': time.time()
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(unified_state, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 State saved to: {self.state_file}")
    
    def load_tree_state(self) -> Optional[Dict[str, Any]]:
        """Load tree merging state from file if it exists."""
        if self.args.state_file and Path(self.args.state_file).exists():
            self.state_file = Path(self.args.state_file)
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Handle different formats
                if 'format_version' in state:
                    # New format
                    print(f"  📂 State loaded (v{state['format_version']}): {self.state_file}")
                    return state
                elif 'completed_tasks' in state:
                    # New format without version
                    print(f"  📂 State loaded (new format): {self.state_file}")
                    return state
                else:
                    # Old format - convert to new format
                    print(f"  📂 State loaded (old format, converting): {self.state_file}")
                    converted_state = {
                        'format_version': '2.0',
                        'current_level': state.get('level', 0),
                        'completed_tasks': {},
                        'legacy_data': state  # Keep old data for reference
                    }
                    return converted_state
                    
            except json.JSONDecodeError as e:
                print(f"  ❌ Error reading state file: {e}")
                print(f"  🔄 Starting fresh...")
                return None
        return None
    
    def update_task_state(self, level: int, task_index: int, output_file: str):
        """Update state file immediately after a task completes."""
        with self.state_lock:
            if not self.tree_state:
                self.tree_state = {
                    'current_level': level,
                    'completed_tasks': {},
                    'level_files': {}
                }
            
            # Track completed tasks per level
            level_key = str(level)
            if level_key not in self.tree_state['completed_tasks']:
                self.tree_state['completed_tasks'][level_key] = {}
            
            self.tree_state['completed_tasks'][level_key][str(task_index)] = output_file
            self.tree_state['current_level'] = level
            
            # Save updated state
            self.save_tree_state(self.tree_state)
    
    def get_completed_tasks_for_level(self, level: int) -> Dict[int, str]:
        """Get already completed tasks for a specific level."""
        if not self.tree_state:
            return {}
            
        # Handle new format (per-task tracking)
        if 'completed_tasks' in self.tree_state:
            level_key = str(level)
            if level_key not in self.tree_state['completed_tasks']:
                return {}
            # Convert string keys back to integers
            completed = self.tree_state['completed_tasks'][level_key]
            return {int(k): v for k, v in completed.items()}
        
        # Handle old format (level-based tracking) - this is limited but better than nothing
        elif 'level' in self.tree_state and 'current_files' in self.tree_state:
            completed_level = self.tree_state.get('level', 0) - 1  # Previous completed level
            if level < completed_level:
                # This level should be completely done, but we don't know the exact files
                # Fall back to file system detection
                return {}
            elif level == completed_level:
                # Partially completed level - use current_files to infer what's done
                # This is a best-effort attempt
                return {}
        
        return {}
    
    def check_existing_files_for_level(self, level: int, total_tasks: int) -> Dict[int, Path]:
        """Check which files already exist for a level (for resuming)."""
        existing_files = {}
        
        # First check state file
        completed_tasks = self.get_completed_tasks_for_level(level)
        for task_idx, file_path in completed_tasks.items():
            path = Path(file_path)
            if path.exists():
                existing_files[task_idx] = path
                
        # Also check for files that might exist without being in state
        for task_idx in range(total_tasks):
            if task_idx not in existing_files:
                # Try to guess the filename for this task
                # This is a fallback for cases where state was lost but files exist
                base_name = Path(self.args.start).stem
                if '_' in base_name:
                    base_name = base_name.rsplit('_', 1)[0]
                
                # Try common patterns
                possible_names = [
                    f"{base_name}_L{level}_P{task_idx:02d}.md",
                    f"{base_name}_L{level}_{task_idx:03d}.md"
                ]
                
                for name in possible_names:
                    path = self.get_output_path(name)
                    if path.exists():
                        existing_files[task_idx] = path
                        break
                        
        return existing_files
    
    def analyze_existing_files(self):
        """Analyze what files already exist in the output directory."""
        if not self.args.output_dir:
            return
            
        output_dir = Path(self.args.output_dir)
        if not output_dir.exists():
            return
            
        # Find all analysis_L*.md files
        pattern = "analysis_L*.md"
        existing_files = list(output_dir.glob(pattern))
        
        if existing_files:
            print(f"  📁 Found {len(existing_files)} existing merge files:")
            for file in sorted(existing_files):
                print(f"    • {file.name}")
    
    def detect_resume_point(self, original_files: List[Path]) -> Optional[Dict]:
        """Detect resume point by analyzing existing merge files on disk.
        
        Returns dictionary with 'files' (list of paths) and 'level' (int),
        or None if no existing files found.
        """
        if not self.args.output_dir:
            return None
            
        output_dir = Path(self.args.output_dir)
        if not output_dir.exists():
            return None
        
        # Look for analysis_L*.md files
        pattern = "analysis_L*.md"
        existing_files = list(output_dir.glob(pattern))
        
        if not existing_files:
            print(f"  📁 No existing merge files found in {output_dir}")
            return None
        
        # Group files by level
        level_files = {}
        for file in existing_files:
            # Extract level from filename like analysis_L1_T2.md
            match = re.search(r'analysis_L(\d+)', file.name)
            if match:
                level = int(match.group(1))
                if level not in level_files:
                    level_files[level] = []
                level_files[level].append(file)
        
        if not level_files:
            return None
        
        # Determine the highest complete level and what to resume with
        max_complete_level = -1
        
        # Calculate expected counts for each level based on tree algorithm
        # This must match the logic in build_merge_tree exactly
        total_files = len(original_files)
        level_expected_counts = []
        current_count = total_files
        level = 0
        
        while current_count > 1:
            # Simulate the exact logic from build_merge_tree
            tasks_this_level = 0
            remaining = current_count
            
            # Count how many merge tasks we'll have
            while remaining > 0:
                if remaining >= 3 and remaining % 2 == 1:
                    # 3-way merge when odd and at least 3
                    tasks_this_level += 1
                    remaining -= 3
                elif remaining >= 2:
                    # 2-way merge
                    tasks_this_level += 1
                    remaining -= 2
                else:
                    # Single file left (shouldn't happen in normal case)
                    break
            
            level_expected_counts.append((level, tasks_this_level))
            
            # Calculate next level count (results from this level)
            current_count = tasks_this_level
            level += 1
        
        print(f"  📊 Expected file counts per level: {level_expected_counts}")
        print(f"  📁 Found files by level: {dict(sorted([(l, len(files)) for l, files in level_files.items()]))}")
        
        # Find highest complete level
        for level, expected_count in level_expected_counts:
            actual_count = len(level_files.get(level, []))
            if actual_count >= expected_count:
                max_complete_level = level
                print(f"  ✓ Level {level} complete: {actual_count}/{expected_count} files")
            else:
                print(f"  ⏸️  Level {level} incomplete: {actual_count}/{expected_count} files")
                break
        
        # Determine what files to resume with
        if max_complete_level >= 0:
            # Get files from the highest complete level
            next_level_files = sorted(level_files[max_complete_level])
            resume_level = max_complete_level + 1
            
            print(f"  🔄 Will resume from level {resume_level} with {len(next_level_files)} files from level {max_complete_level}")
            print(f"     Files: {[f.name for f in next_level_files[:3]]}{'...' if len(next_level_files) > 3 else ''}")
            
            return {
                'files': next_level_files,
                'level': resume_level
            }
        else:
            # No complete levels found
            print(f"  ❌ No complete levels found, starting fresh")
            return None
    
    def should_skip_to_level(self, target_level: int) -> bool:
        """Check if we should skip directly to a specific level based on existing files."""
        if not self.tree_state or not self.args.output_dir:
            return False
            
        # If state says we're at level 3+, check if we have files for earlier levels
        state_level = self.tree_state.get('level', 0)
        if state_level <= target_level:
            return False
            
        # Check if files exist for levels before target_level
        output_dir = Path(self.args.output_dir)
        for level in range(target_level):
            level_files = list(output_dir.glob(f"analysis_L{level}_*.md"))
            if not level_files:
                return False  # Missing files for this level
                
        return True
    
    def merge_files_parallel(self, files_list: List[Tuple[Path, ...]], level: int) -> List[Optional[Path]]:
        """Merge multiple file pairs/triplets in parallel."""
        results = [None] * len(files_list)
        
        # Check for existing files (resume capability)
        existing_files = self.check_existing_files_for_level(level, len(files_list))
        if existing_files:
            print(f"  ♻️  Found {len(existing_files)} existing files for level {level}, resuming...")
            for task_idx, file_path in existing_files.items():
                results[task_idx] = file_path
                print(f"    ✓ Skipping task {task_idx}: {file_path.name} already exists")
        
        def merge_task(index: int, files_tuple: Tuple[Path, ...]):
            """Task for merging a single set of files."""
            try:
                if len(files_tuple) == 3:
                    # 3-way merge
                    result = self.merge_three_files(files_tuple[0], files_tuple[1], files_tuple[2], level, index)
                else:
                    # 2-way merge
                    result = self.merge_two_files_tree(files_tuple[0], files_tuple[1], level, index)
                
                # Save immediately after successful merge
                if result is not None:
                    output_filename = self.get_tree_output_filename(level, index, files_tuple)
                    output_path = self.get_output_path(output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result)
                    
                    # Update state immediately after each successful merge
                    self.update_task_state(level, index, str(output_path))
                    
                    print(f"    💾 Task {index}: Saved {output_path.name} ({len(result)} chars)")
                    return index, output_path  # Return the saved file path instead of content
                else:
                    return index, None
                    
            except Exception as e:
                print(f"  ❌ Error in parallel merge {index}: {str(e)}")
                return index, None
        
        # Only submit tasks that aren't already completed
        tasks_to_run = [(i, files_tuple) for i, files_tuple in enumerate(files_list) if results[i] is None]
        
        if tasks_to_run:
            print(f"  🚀 Running {len(tasks_to_run)} remaining tasks...")
            
            with ThreadPoolExecutor(max_workers=min(self.args.max_parallel, len(tasks_to_run))) as executor:
                futures = []
                for i, files_tuple in tasks_to_run:
                    # Small delay between submissions to spread out API calls
                    if len(futures) > 0:
                        time.sleep(0.2)  # 200ms delay between submissions
                    future = executor.submit(merge_task, i, files_tuple)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    index, result = future.result()
                    results[index] = result
        else:
            print(f"  ✓ All tasks for level {level} already completed!")
                
        return results
    
    def merge_two_files_tree(self, file1: Path, file2: Path, level: int, pair_num: int) -> Optional[str]:
        """Merge two files for tree-based merging."""
        print(f"    📝 Level {level}, Pair {pair_num + 1}: {file1.name} + {file2.name}")
        
        # Load file contents
        with open(file1, 'r', encoding='utf-8') as f:
            content1 = f.read()
        with open(file2, 'r', encoding='utf-8') as f:
            content2 = f.read()
        
        print(f"      File 1: {len(content1)} chars")
        print(f"      File 2: {len(content2)} chars")
        
        # Prepare prompt
        prompt = self.prompt_template.replace('{base_file_content}', content1)
        prompt = prompt.replace('{merge_file_content}', content2)
        
        # Call API (parallel in tree mode)  
        result = self.call_openrouter_api(prompt)
        
        if result and self.validate_merge_result(content1, result, content2):
            return result
        
        return None
    
    def merge_three_files(self, file1: Path, file2: Path, file3: Path, level: int, pair_num: int) -> Optional[str]:
        """Merge three files (for handling odd numbers)."""
        print(f"    📝 Level {level}, Triplet {pair_num + 1}: {file1.name} + {file2.name} + {file3.name}")
        
        # First merge file1 and file2
        with open(file1, 'r', encoding='utf-8') as f:
            content1 = f.read()
        with open(file2, 'r', encoding='utf-8') as f:
            content2 = f.read()
        
        print(f"      Step 1: Merging first two files ({len(content1)} + {len(content2)} chars)")
        
        prompt = self.prompt_template.replace('{base_file_content}', content1)
        prompt = prompt.replace('{merge_file_content}', content2)
        
        intermediate = self.call_openrouter_api(prompt)
        
        if not intermediate:
            return None
        
        # Then merge with file3
        with open(file3, 'r', encoding='utf-8') as f:
            content3 = f.read()
        
        print(f"      Step 2: Merging with third file ({len(intermediate)} + {len(content3)} chars)")
        
        prompt = self.prompt_template.replace('{base_file_content}', intermediate)
        prompt = prompt.replace('{merge_file_content}', content3)
        
        result = self.call_openrouter_api(prompt)
        
        # For 3-way merge, validate against the total size of all inputs
        if result:
            # Special validation for 3-way merge
            total_input = len(content1) + len(content2) + len(content3)
            merged_size = len(result)
            avg_size = total_input / 3
            min_acceptable_size = int(avg_size * 0.9)
            
            if merged_size >= min_acceptable_size:
                print(f"  ✓ 3-way validation passed: {int(avg_size)} avg → {merged_size} chars")
                return result
            else:
                print(f"  ❌ 3-way validation failed: Output ({merged_size}) smaller than acceptable ({min_acceptable_size})")
        
        return None
    
    def get_output_path(self, filename: str) -> Path:
        """Get the output file path."""
        if self.args.output_dir:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / filename
        else:
            return Path(filename)
            
    def run_tree_merge(self, files: List[Path]) -> Optional[Path]:
        """Execute tree-based merging."""
        print("🌳 Tree-based merging strategy")
        print("-" * 50)
        
        # Ensure output directory exists
        if self.args.output_dir:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory: {output_dir.absolute()}")
        
        # Load state if resuming
        if self.args.state_file:
            self.tree_state = self.load_tree_state()
        else:
            # Auto-detect state file
            base_name = Path(self.args.start).stem
            if '_' in base_name:
                base_name = base_name.rsplit('_', 1)[0]
            auto_state_file = self.get_output_path(f"{base_name}_tree_state.json")
            if auto_state_file.exists():
                print(f"  📂 Found existing state file: {auto_state_file}")
                self.state_file = auto_state_file
                self.tree_state = self.load_tree_state()
                
        # If we have state, analyze what's been completed
        if self.tree_state:
            if 'level' in self.tree_state:
                resume_level = self.tree_state.get('level', 0)
                print(f"  🔄 Resuming from level {resume_level}")
                
                # Check what files exist on disk
                self.analyze_existing_files()
        
        # Calculate expected levels for display
        total_files = len(files)
        print(f"📊 Tree merge: {total_files} files")
        print()
        
        base_name = Path(self.args.start).stem
        if '_' in base_name:
            base_name = base_name.rsplit('_', 1)[0]
        
        # Determine starting point - check file system first, then state
        resume_info = self.detect_resume_point(files)
        
        if self.tree_state:
            # Handle legacy format
            if 'legacy_data' in self.tree_state and 'current_files' in self.tree_state['legacy_data']:
                legacy_data = self.tree_state['legacy_data']
                state_level = legacy_data.get('level', 0)
                state_files = [Path(f) for f in legacy_data['current_files'] if Path(f).exists()]
                
                if state_files and state_level > 0:
                    print(f"  🔄 Resuming from legacy state: level {state_level} with {len(state_files)} files")
                    current_files = state_files
                    level_idx = state_level
                elif resume_info:
                    print(f"  🔄 Legacy state files missing, using file system detection")
                    current_files = resume_info['files']
                    level_idx = resume_info['level']
                else:
                    current_files = files.copy()
                    level_idx = 0
            # Handle new format
            elif 'current_level' in self.tree_state:
                if resume_info:
                    print(f"  🔄 Using file system detection for resume")
                    current_files = resume_info['files']
                    level_idx = resume_info['level']
                else:
                    current_files = files.copy()
                    level_idx = 0
            else:
                if resume_info:
                    current_files = resume_info['files']
                    level_idx = resume_info['level']
                else:
                    current_files = files.copy()
                    level_idx = 0
        else:
            # No state file - use file system detection
            if resume_info:
                print(f"  🔄 No state file, using file system detection")
                current_files = resume_info['files']
                level_idx = resume_info['level']
            else:
                current_files = files.copy()
                level_idx = 0
        
        # Process levels iteratively
        while len(current_files) > 1:
            print(f"\n{'=' * 60}")
            print(f"🌳 Processing Level {level_idx}")
            print(f"{'=' * 60}")
            
            # Create groups for this level
            groups_to_process = []
            i = 0
            while i < len(current_files):
                remaining = len(current_files) - i
                
                if remaining >= 3 and remaining % 2 == 1:
                    # If odd number of items left and at least 3, merge three
                    groups_to_process.append((current_files[i], current_files[i + 1], current_files[i + 2]))
                    i += 3
                elif remaining >= 2:
                    # If even number or exactly 2 left, merge pair
                    groups_to_process.append((current_files[i], current_files[i + 1]))
                    i += 2
                else:
                    # Single item left - shouldn't happen with proper logic above
                    break
            
            # Merge groups in parallel
            results = self.merge_files_parallel(groups_to_process, level_idx)
            
            # Check for failures
            failed_groups = [i for i, result in enumerate(results) if result is None]
            if failed_groups:
                print(f"\n❌ Failed merges at level {level_idx}: groups {failed_groups}")
                # Save current state
                state = {
                    'current_level': level_idx,
                    'completed_tasks': self.tree_state.get('completed_tasks', {}) if self.tree_state else {},
                    'failed_groups': failed_groups
                }
                self.save_tree_state(state)
                return None
            
            # Collect successful results (files are already saved)
            next_level_files = []
            successful_results = [r for r in results if r is not None]
            print(f"  📊 Level {level_idx}: {len(successful_results)}/{len(results)} tasks successful")
            
            for i, result_path in enumerate(results):
                if result_path is None:
                    print(f"  ⚠️  Task {i} failed - skipping")
                    continue
                next_level_files.append(result_path)
            
            if not next_level_files:
                print(f"  ❌ No files saved at level {level_idx} - all tasks failed")
                return None
            
            current_files = next_level_files
            
            # Save state after successful level (always save for tree mode)
            state = {
                'current_level': level_idx + 1,
                'completed_tasks': self.tree_state.get('completed_tasks', {}) if self.tree_state else {}
            }
            self.save_tree_state(state)
            
            # Move to next level
            level_idx += 1
        
        # Final result should be a single file
        if len(current_files) == 1:
            final_output = self.get_output_path(f"{base_name}_tree_merged_final.md")
            
            # Copy or rename the final file
            import shutil
            shutil.copy2(current_files[0], final_output)
            
            print(f"\n{'=' * 60}")
            print(f"✅ Tree merge completed successfully!")
            print(f"   Final output: {final_output}")
            
            # Cleanup intermediate files if not saving them
            if not self.args.save_intermediate:
                print("🧹 Cleaning up intermediate files...")
                # Find all intermediate files created during tree merge
                intermediate_pattern = f"{base_name}_L*.md"
                import glob
                for pattern_file in glob.glob(str(self.get_output_path(intermediate_pattern))):
                    intermediate_file = Path(pattern_file)
                    if intermediate_file != final_output and intermediate_file.exists():
                        intermediate_file.unlink()
                        print(f"    🗑️  Removed: {intermediate_file.name}")
                        
            # Cleanup state file
            if self.state_file and self.state_file.exists():
                self.state_file.unlink()
                
            return final_output
        else:
            print(f"\n❌ Tree merge failed: Expected 1 final file, got {len(current_files)}")
            return None

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
        
        # Try merge with validation retries
        merge_attempts = 0
        max_merge_attempts = 3
        
        while merge_attempts < max_merge_attempts:
            merge_attempts += 1
            
            if merge_attempts > 1:
                print(f"  🔄 Retrying merge (attempt {merge_attempts}/{max_merge_attempts})...")
                time.sleep(self.retry_delay)
            
            # Call API
            result = self.call_openrouter_api(prompt)
            
            if result is None:
                print("  ❌ API call failed after all retries")
                continue
                
            # Validate result
            if self.validate_merge_result(base_content, result):
                return result
            else:
                print(f"  ⚠️  Validation failed, will retry..." if merge_attempts < max_merge_attempts else "")
        
        print("  ❌ Merge failed after all attempts")
        return None
        
    def run(self):
        """Main execution flow."""
        print("🚀 File Merger Starting")
        print("=" * 60)
        
        # Discover files
        files = self.discover_files()
        if len(files) < 2:
            print("❌ Error: Need at least 2 files to merge")
            sys.exit(1)
            
        # Handle different strategies
        if self.args.merge_strategy == 'tree':
            # Tree merging doesn't use base_file concept the same way
            if self.args.base_file:
                print("⚠️  Warning: --base-file is not used with tree merging strategy")
                
            if not self.show_files_for_confirmation(files):
                print("❌ Merge cancelled by user")
                sys.exit(0)
                
            # Load prompt template
            self.prompt_template = self.load_prompt_template()
            print(f"✓ Prompt template loaded from: {self.args.prompt_file}")
            
            # Run tree merge
            result = self.run_tree_merge(files)
            if result:
                print(f"\n✅ Tree merge completed: {result}")
            else:
                print("\n❌ Tree merge failed")
                sys.exit(1)
            return
            
        # Sequential merging (original logic)
        if self.args.base_file:
            print(f"📌 Using explicit base file: {self.args.base_file}")
            base_file = Path(self.args.base_file)
            if not base_file.exists():
                print(f"❌ Error: Base file not found: {base_file}")
                sys.exit(1)
            # Remove base file from list if it's there
            files = [f for f in files if f != base_file]
        else:
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
    parser.add_argument('--merge-strategy', choices=['sequential', 'tree'], default='sequential',
                        help='Merge strategy: sequential (default) or tree-based')
    parser.add_argument('--max-parallel', type=int, default=3,
                        help='Maximum parallel API calls for tree merging (default: 3)')
    parser.add_argument('--state-file', help='State file for resuming tree merges (auto-generated if not specified)')
    
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