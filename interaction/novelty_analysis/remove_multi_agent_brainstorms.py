import json
import os
from pathlib import Path
from collections import defaultdict
import argparse

def process_jsonl_file(filepath):
    """Process a single JSONL file to remove first 2 of every 4 duplicate round entries."""
    
    # Read all lines from the file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse JSON and group by round number
    parsed_lines = []
    for line in lines:
        if line.strip():  # Skip empty lines
            try:
                data = json.loads(line)
                parsed_lines.append((data, line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {filepath}: {line[:50]}...")
    
    # Group lines by round number
    round_groups = defaultdict(list)
    for data, original_line in parsed_lines:
        if 'round' in data:
            round_groups[data['round']].append((data, original_line))
        else:
            # If no round field, keep the line as-is
            round_groups[None].append((data, original_line))
    
    # Process groups and build output
    output_lines = []
    for round_num in sorted(round_groups.keys(), key=lambda x: (x is None, x)):
        group = round_groups[round_num]
        
        if round_num is not None and len(group) == 4:
            # For groups of exactly 4 with same round, keep only the last 2
            output_lines.extend([line for _, line in group[-2:]])
        else:
            # Keep all lines for other cases
            output_lines.extend([line for _, line in group])
    
    # Write back to the file
    filepath = Path(filepath)
    folder = filepath.parent / 'processed'
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / filepath.name, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"Processed {filepath}: {len(lines)} lines -> {len(output_lines)} lines")

def remove_multi_agent_brainstorms(folder_path='../conversation_data/novelty1/temp0.7/1'):
    # Find all .jsonl files
    jsonl_files = list(Path(folder_path).glob('*.jsonl'))
    
    if not jsonl_files:
        print("No .jsonl files found in the current directory")
        return
    
    print(f"Found {len(jsonl_files)} .jsonl file(s)")
    
    # Process each file
    for filepath in jsonl_files:
        print(f"\nProcessing {filepath}...")
        process_jsonl_file(filepath)
    
    print("\nAll files processed successfully!")
