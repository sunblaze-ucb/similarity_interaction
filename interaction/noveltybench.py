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
        "final": "Now, write the best story in five sentences about a girl and her dog.",
    },
    "biography": {
        "folder_idx": 2,
        "brainstorm": (
            "Brainstorm a short biography for an imaginary historical figure, "
            "including their birth and death dates, profession, and greatest contribution."
        ),
        "final": (
            "Now, write a short biography of a fictional historical figure inspired by the "
            "brainstorming results. Your bio should include their birth and death dates, "
            "profession, and greatest contribution."
        ),
    },
    "haiku": {
        "folder_idx": 3,
        "brainstorm": "Brainstorm a plot for a haiku about a whale and a walnut tree.",
        "final": "Now, write the best haiku about a whale and a walnut tree.",
    },
    "vacation": {
        "folder_idx": 4,
        "brainstorm": "Brainstorm some good points about going on vacation.",
        "final": (
            "Now, please respond to this question: What's the one best thing about going on a vacation?"
        ),
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a multi-agent novelty-bench style brainstorm+write task across model pairs."
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


def cleanup(model1=None, model2=None):
    if model1 is not None:
        del model1
    if model2 is not None:
        del model2
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
        / f"novelty{task_cfg['folder_idx']}"
        / f"temp{args.temperature}"
        / "1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_files = os.listdir(out_dir)

    model_list = models.model_config.MODEL_LIST

    for model_id1 in model_list:
        for model_id2 in model_list:
            print(f"{model_id1}, {model_id2} conversation starts!")
            model_name1 = model_id1[model_id1.index("/") + 1:]
            model_name2 = model_id2[model_id2.index("/") + 1:]
            outfile = out_dir / f"{model_name1}_{model_name2}.jsonl"
            done = {f"{model_name1}_{model_name2}.jsonl", f"{model_name2}_{model_name1}.jsonl"}
            if done & set(existing_files):
                continue

            model1 = model2 = None
            try:
                model1 = Model(model_id1, args.temperature)
                if model_id1 != model_id2 or args.temperature > 0:
                    model2 = Model(model_id2, args.temperature)
                existing_files.append(f"{model_name1}_{model_name2}.jsonl")

                with open(outfile, "w", encoding="utf-8") as f:
                    for r in tqdm(range(args.rounds)):
                        history1, history2 = [], []

                        m1_resp = converse(history1, model1, brainstorm_prompt, f, r, model_id1)
                        if model2 is not None:
                            m2_resp = converse(history2, model2, brainstorm_prompt, f, r, model_id2)
                        else:
                            m2_resp = m1_resp
                            f.write(json.dumps({
                                "round": r,
                                "speaker": model_id2,
                                "input": brainstorm_prompt,
                                "text": m2_resp,
                            }, ensure_ascii=False) + "\n")
                            f.flush()

                        combined1 = (
                            f"This is a combined result of your brainstorming and the other's brainstorming:\n"
                            f"{m1_resp}\n{m2_resp}\n\n{final_prompt}"
                        )
                        combined2 = (
                            f"This is a combined result of your brainstorming and the other's brainstorming:\n"
                            f"{m2_resp}\n{m1_resp}\n\n{final_prompt}"
                        )
                        converse(history1, model1, combined1, f, r, model_id1)
                        converse(history2, model2 if model2 is not None else model1, combined2, f, r, model_id2)

                cleanup(model1, model2)
            except Exception as e:
                print(e)
                cleanup(model1, model2)
                if outfile.exists():
                    outfile.unlink()
                continue


if __name__ == "__main__":
    main()
