import json
import os
from pathlib import Path
import argparse

def process_jsonl_file(filepath):
    """Process a single .jsonl file to remove duplicate rounds."""
    
    # Read all lines from the file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse each line and store with its round number
    entries = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            try:
                data = json.loads(line)
                entries.append((data.get('round'), line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {filepath}: {line[:50]}...")
                continue
    
    # Keep only the last occurrence of each round number
    seen_rounds = {}
    for round_num, line in entries:
        seen_rounds[round_num] = line  # This overwrites previous occurrences
    
    # Get the filtered lines in the original order (by round number)
    filtered_lines = []
    unique_rounds = sorted(seen_rounds.keys())
    for round_num in unique_rounds:
        filtered_lines.append(seen_rounds[round_num])
    
    filepath = Path(filepath)
    folder = filepath.parent / 'processed'
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / filepath.name, 'w', encoding='utf-8') as f:
        for line in filtered_lines:
            f.write(line + '\n')
    
    print(f"Processed {filepath.name}: {len(entries)} lines -> {len(filtered_lines)} lines")

def remove_single_agent_brainstorm(folder_path='../conversation_data/single_novelty1/temp0.7/1'):
    
    # Find all .jsonl files
    jsonl_files = list(Path(folder_path).glob('*.jsonl'))
    
    if not jsonl_files:
        print("No .jsonl files found in the current directory")
        return
    
    print(f"Found {len(jsonl_files)} .jsonl file(s)")
    
    # Process each file
    for jsonl_file in jsonl_files:
        process_jsonl_file(jsonl_file)
    
    print("\nAll files processed!")