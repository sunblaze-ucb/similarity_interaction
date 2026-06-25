#!/usr/bin/env python3
"""
Script to add 'single_agent_text' field to multiagent data
by randomly selecting from corresponding single agent data.
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
import glob
import argparse

def load_single_agent_data(single_agent_path):
    """
    Load all single agent data into a dictionary organized by model name.
    Returns: dict where keys are model names and values are lists of text entries
    """
    single_agent_texts = defaultdict(list)
    
    # Get all .jsonl files in single agent directory
    single_agent_files = glob.glob(os.path.join(single_agent_path, "*.jsonl"))
    
    print(f"Loading single agent data from {len(single_agent_files)} files...")
    
    for filepath in single_agent_files:
        # Extract model name from filename (remove .jsonl extension)
        model_name = os.path.basename(filepath).replace('.jsonl', '')
        
        # Read all lines from the file
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        single_agent_texts[model_name].append(data['text'])
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {filepath}: {e}")
                    continue
        
        print(f"  Loaded {len(single_agent_texts[model_name])} texts for model: {model_name}")
    
    return single_agent_texts

def process_multiagent_files(multiagent_path, single_agent_texts, output_path=None):
    """
    Process all multiagent files and add single_agent_text field.
    """
    # Get all .jsonl files in multiagent directory
    multiagent_files = glob.glob(os.path.join(multiagent_path, "*.jsonl"))
    
    print(f"\nProcessing {len(multiagent_files)} multiagent files...")
    
    # Create output directory if specified and doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    
    for filepath in multiagent_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")
        
        # Determine output file path
        if output_path:
            output_filepath = os.path.join(output_path, filename)
        else:
            # Overwrite original file (make backup first if desired)
            output_filepath = filepath
            backup_filepath = filepath + '.backup'
            os.rename(filepath, backup_filepath)
            print(f"  Created backup: {backup_filepath}")
        
        processed_lines = []
        missing_speakers = set()
        
        # Read and process each line
        with open(filepath if not output_path else filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Get the speaker field
                    if 'speaker' in data:
                        # strip of / and everything before it
                        speaker = data['speaker']
                        if '/' in speaker:
                            speaker = speaker.split('/')[-1]
                        
                        # Check if we have single agent data for this speaker
                        if speaker in single_agent_texts and single_agent_texts[speaker]:
                            # Randomly select a text from the single agent data
                            random_text = random.choice(single_agent_texts[speaker])
                            data['single_agent_text'] = random_text
                        else:
                            # Speaker not found in single agent data
                            missing_speakers.add(speaker)
                            data['single_agent_text'] = None  # or you can skip adding this field
                    
                    processed_lines.append(json.dumps(data, ensure_ascii=False))
                    
                except json.JSONDecodeError as e:
                    print(f"  Error parsing line {line_num} in {filename}: {e}")
                    # Keep the original line if there's an error
                    processed_lines.append(line.strip())
        
        # Write processed data to output file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
        
        print(f"  Processed {len(processed_lines)} lines")
        if missing_speakers:
            print(f"  Warning: Could not find single agent data for speakers: {missing_speakers}")

def add_single_agent_text(single_agent_path='../conversation_data/single_novelty1/temp0.7/1/processed', multiagent_path='../conversation_data/novelty1/temp0.7/1/processed', output_path='../conversation_data/novelty1/analysis/MI/temp0.7/1'):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Adding single_agent_text field to multiagent data")
    print("="*60)
    
    # Load all single agent data
    single_agent_texts = load_single_agent_data(single_agent_path)
    
    if not single_agent_texts:
        print("Error: No single agent data loaded!")
        return
    
    print(f"\nTotal models loaded: {len(single_agent_texts)}")
    print(f"Models available: {list(single_agent_texts.keys())}")
    
    # Process multiagent files
    # Set OUTPUT_PATH to None if you want to overwrite original files (with backup)
    # Or specify a different path to save modified files separately
    process_multiagent_files(multiagent_path, single_agent_texts, output_path)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)