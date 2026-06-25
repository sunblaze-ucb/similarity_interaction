import json
import os
import glob
import torch
from pathlib import Path
from typing import Dict, List
import numpy as np
import argparse

# file processing
from remove_multi_agent_brainstorms import remove_multi_agent_brainstorms
from remove_single_agent_brainstorm import remove_single_agent_brainstorm
from add_single_agent_text import add_single_agent_text

# LLM Entropy Calculator
from llm_vocab_entropy import LLMEntropyCalculator

def calculate_conditional_cross_entropy(calculator, context_text: str, target_text: str) -> float:
    """
    Calculate H_theta(target_text | context_text)
    This is the cross-entropy of target_text given context_text as conditioning
    """
    # Concatenate context and target
    full_text = context_text + target_text
    
    # Tokenize to get lengths
    context_tokens = calculator.tokenizer(context_text, return_tensors="pt")
    full_tokens = calculator.tokenizer(full_text, return_tensors="pt", truncation=True)
    
    context_length = context_tokens["input_ids"].shape[1]
    
    # Move to model device
    input_ids = full_tokens["input_ids"].to(calculator.model_device)
    attention_mask = full_tokens["attention_mask"].to(calculator.model_device) if "attention_mask" in full_tokens else None
    
    # Get model outputs
    with torch.no_grad():
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        outputs = calculator.model(**model_inputs)
        logits = outputs.logits  # [batch, seq_len, vocab]
    
    # Calculate cross-entropy only for the target portion
    total_ce = 0.0
    token_count = 0
    
    # Start from context_length-1 because position i predicts token i+1
    for i in range(context_length - 1, input_ids.shape[1] - 1):
        logits_at_i = logits[0, i, :]
        actual_next_token = input_ids[0, i + 1]
        
        # Get log probability of the actual next token
        log_probs = torch.nn.functional.log_softmax(logits_at_i, dim=-1)
        token_ce = -log_probs[actual_next_token].item()
        
        total_ce += token_ce
        token_count += 1
    
    # Return mean cross-entropy over target tokens
    return total_ce / token_count if token_count > 0 else 0.0

def process_jsonl_file(file_path: str, calculator: LLMEntropyCalculator, output_dir: str = None):
    """
    Process a single JSONL file and add mutual information field to each line
    """
    print(f"Processing: {file_path}")
    
    # Read the file
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                entries.append(json.loads(line))
    
    # Process each entry
    for idx, entry in enumerate(entries):
        if idx % 10 == 0:
            print(f"  Processing entry {idx+1}/{len(entries)}")
        
        text = entry.get("text", "")
        single_agent_text = entry.get("single_agent_text", "")
        
        if not text or not single_agent_text:
            print(f"  Warning: Entry {idx} missing required fields, skipping")
            entry["mutual_information_nats"] = None
            entry["mutual_information_bits"] = None
            continue
        
        try:
            # Calculate H_theta(text) - unconditional cross-entropy
            h_text = calculator.calculate_utterance_cross_entropy(text)["mean_cross_entropy_nats"]
            
            # Calculate H_theta(text | single_agent_text) - conditional cross-entropy
            h_text_given_single = calculate_conditional_cross_entropy(
                calculator, single_agent_text, text
            )
            
            # Calculate mutual information: MI = H(text) - H(text|single_agent_text)
            mutual_info_nats = h_text - h_text_given_single
            mutual_info_bits = mutual_info_nats / np.log(2)
            
            # Add to entry
            entry["mutual_information_nats"] = mutual_info_nats
            entry["mutual_information_bits"] = mutual_info_bits
            entry["h_text_nats"] = h_text
            entry["h_text_given_single_nats"] = h_text_given_single
            
        except Exception as e:
            print(f"  Error processing entry {idx}: {e}")
            entry["mutual_information_nats"] = None
            entry["mutual_information_bits"] = None
            entry["error"] = str(e)
    
    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(file_path))
    else:
        # Save with a suffix to avoid overwriting original
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_with_mi{ext}"
    
    # Write updated entries back to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"  Saved to: {output_path}")
    
    # Print summary statistics
    valid_mis = [e["mutual_information_nats"] for e in entries 
                 if e.get("mutual_information_nats") is not None]
    if valid_mis:
        print(f"  Mutual Information Statistics (nats):")
        print(f"    Mean: {np.mean(valid_mis):.4f}")
        print(f"    Std:  {np.std(valid_mis):.4f}")
        print(f"    Min:  {np.min(valid_mis):.4f}")
        print(f"    Max:  {np.max(valid_mis):.4f}")

def main():
    # folder path and output directory
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature",
        default="0.7"
    )
    parser.add_argument("--novelty_task", default="1")
    repeat = 1
    
    args = parser.parse_args()
    temperature = args.temperature
    novelty_task = args.novelty_task

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Or "meta-llama/Llama-3.1-8B"
    
    # Initialize the calculator
    print(f"Initializing model: {MODEL_NAME}")
    calculator = LLMEntropyCalculator(model_name=MODEL_NAME)
    
    single_agent_path_original = f'../conversation_data/single_novelty{novelty_task}/temp{temperature}/{repeat}'
    multi_agent_path_original = f'../conversation_data/novelty{novelty_task}/temp{temperature}/{repeat}'
    single_agent_path=single_agent_path_original + '/processed'
    multiagent_path=multi_agent_path_original + '/processed'
    output_path=f'../conversation_data/novelty{novelty_task}/analysis/MI/temp{temperature}/{repeat}'
    
    remove_single_agent_brainstorm(single_agent_path_original)
    remove_multi_agent_brainstorms(multi_agent_path_original)
    add_single_agent_text(single_agent_path, multiagent_path, output_path)

    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(output_path, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {output_path}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Process each file
    for file_path in jsonl_files:
        try:
            process_jsonl_file(file_path, calculator, output_path)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue
    
    print("\nAll files processed!")

if __name__ == "__main__":
    main()