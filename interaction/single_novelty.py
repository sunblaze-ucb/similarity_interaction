import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, TextIO

import torch
import torch._dynamo as dynamo
from tqdm import tqdm

from models.model import Model
import models.model_config


TASK_PROMPTS = {
    "story": {
        "folder_idx": 1,
        "brainstorm": "Brainstorm a plot for a story about a girl and her dog.",
        "final": "Based on your brainstorming, write the best story in five sentences about a girl and her dog.",
    },
    "biography": {
        "folder_idx": 2,
        "brainstorm": (
            "Brainstorm a short biography for an imaginary historical figure, "
            "including their birth and death dates, profession, and greatest contribution."
        ),
        "final": (
            "Now, write a short biography of a fictional historical figure inspired by your "
            "brainstorming. Your bio should include their birth and death dates, profession, "
            "and greatest contribution."
        ),
    },
    "haiku": {
        "folder_idx": 3,
        "brainstorm": "Brainstorm a plot for a haiku about a whale and a walnut tree.",
        "final": "Now, based on your brainstorming, write the best haiku about a whale and a walnut tree.",
    },
    "vacation": {
        "folder_idx": 4,
        "brainstorm": "Brainstorm some good points about going on vacation.",
        "final": (
            "Now, based on your brainstorming, please respond to this question: "
            "What's the one best thing about going on a vacation?"
        ),
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a single-agent novelty brainstorm+write task across models."
    )
    p.add_argument("--task", required=True, choices=list(TASK_PROMPTS.keys()))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--output-root", default="conversation_data")
    return p.parse_args()


def converse(history: List[Dict[str, str]], model: Model, prompt: str,
             log_file: TextIO, round_idx: int, speaker: str) -> str:
    history.append({"role": "user", "content": prompt})
    reply = model.generate_text([history])[0]
    history.append({"role": "assistant", "content": reply})
    log_file.write(json.dumps({
        "round": round_idx,
        "speaker": speaker,
        "input": prompt,
        "text": reply,
    }, ensure_ascii=False) + "\n")
    log_file.flush()
    return reply


def cleanup(model=None):
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    dynamo.reset()
    torch.cuda.ipc_collect()


def main():
    args = parse_args()
    task_cfg = TASK_PROMPTS[args.task]
    brainstorm_prompt = task_cfg["brainstorm"]
    final_prompt = task_cfg["final"]

    out_dir = (
        Path(args.output_root)
        / f"single_novelty{task_cfg['folder_idx']}"
        / f"temp{args.temperature}"
        / "1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_files = os.listdir(out_dir)

    model_list = models.model_config.MODEL_LIST

    for model_id in model_list:
        print(f"{model_id} starts!")
        model_name = model_id[model_id.index("/") + 1:]
        outfile = out_dir / f"{model_name}.jsonl"
        if f"{model_name}.jsonl" in existing_files:
            continue

        model = None
        try:
            model = Model(model_id, args.temperature)
            existing_files.append(f"{model_name}.jsonl")

            with open(outfile, "w", encoding="utf-8") as f:
                for r in tqdm(range(args.rounds)):
                    history = []
                    converse(history, model, brainstorm_prompt, f, r, model_id)
                    converse(history, model, final_prompt, f, r, model_id)

            cleanup(model)
        except Exception as e:
            print(e)
            cleanup(model)
            if outfile.exists():
                outfile.unlink()
            continue


if __name__ == "__main__":
    main()
