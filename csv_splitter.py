#!/usr/bin/env python3
"""
CSV Splitter for Telegram Parser
Splits large CSV files from telegram_parser.py into smaller files.
"""

import csv
import argparse
from pathlib import Path
from typing import Optional


def split_csv(
    input_file: Path,
    split_size: int,
    output_dir: Optional[Path] = None,
    prefix: str = "messages"
) -> list[Path]:
    """
    Split a CSV file into multiple smaller files.
    
    Args:
        input_file: Path to the input CSV file
        split_size: Maximum number of rows per split file (excluding header)
        output_dir: Directory to save split files (defaults to same as input file)
        prefix: Prefix for output files (default: "messages")
    
    Returns:
        List of created file paths
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_dir is None:
        output_dir = input_file.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    current_file_index = 1
    current_row_count = 0
    headers = None
    current_writer = None
    current_file = None
    
    print(f"Splitting {input_file} into files with {split_size} messages each...")
    
    try:
        with open(input_file, 'r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            
            # Read headers
            headers = next(reader)
            
            for row in reader:
                # Open new file if needed
                if current_row_count == 0:
                    if current_file:
                        current_file.close()
                    
                    output_file = output_dir / f"{prefix}_{current_file_index:03d}.csv"
                    created_files.append(output_file)
                    current_file = open(output_file, 'w', encoding='utf-8', newline='')
                    current_writer = csv.writer(current_file)
                    
                    # Write headers
                    current_writer.writerow(headers)
                    
                    print(f"Creating: {output_file.name}")
                
                # Write row
                current_writer.writerow(row)
                current_row_count += 1
                
                # Check if we need to move to next file
                if current_row_count >= split_size:
                    current_file_index += 1
                    current_row_count = 0
    
    finally:
        if current_file:
            current_file.close()
    
    return created_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split large CSV files from telegram_parser.py into smaller files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s messages.csv --split-size 1000        # Split into 1000-message files
  %(prog)s messages.csv -s 500 -o ./split        # Split into 500-message files in ./split/
  %(prog)s messages.csv -s 2000 --prefix export  # Use 'export' prefix instead of 'messages'
"""
    )
    
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input CSV file to split"
    )
    parser.add_argument(
        "-s", "--split-size",
        type=int,
        required=True,
        help="Maximum number of messages per split file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory for split files (default: same as input file)"
    )
    parser.add_argument(
        "--prefix",
        default="messages",
        help="Prefix for output files (default: messages)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.split_size <= 0:
        print("Error: --split-size must be a positive integer")
        return 1
    
    try:
        created_files = split_csv(
            args.input_file,
            args.split_size,
            args.output_dir,
            args.prefix
        )
        
        print(f"\n✓ Split complete!")
        print(f"  Input file: {args.input_file.absolute()}")
        print(f"  Created {len(created_files)} files:")
        for file_path in created_files:
            print(f"    {file_path.absolute()}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(exit_code := main())