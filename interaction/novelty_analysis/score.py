import argparse
import asyncio
import bisect
import functools
import json
import os
from pathlib import Path

import datasets
import numpy as np
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CONCURRENT_REQUESTS = 1

TASK_CONFIG = {
    "story": {
        "folder_idx": 1,
        "reference_prompt": "Tell me a story in five sentences about a girl and her dog.",
    },
    "biography": {
        "folder_idx": 2,
        "reference_prompt": (
            "Your task is to write a short biography for a made-up historic figure. Your bio "
            "should include their birth and death dates, profession, and greatest contribution."
        ),
    },
    "haiku": {
        "folder_idx": 3,
        "reference_prompt": "Write a haiku about a whale and a walnut tree.",
    },
    "vacation": {
        "folder_idx": 4,
        "reference_prompt": "What's the one best thing about going on a vacation?",
    },
}

reward_thresholds = [
    -7.71875,
    -6.28125,
    -6.0,
    -5.71875,
    -5.5,
    -5.0,
    -4.375,
    -3.4375,
    -2.046875,
]


def transform_raw_reward(reward: float) -> int:
    return bisect.bisect_left(reward_thresholds, reward) + 1


@functools.cache
def rm_and_tokenizer():
    model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, tokenizer


@torch.inference_mode()
async def score_partition_rm(prompt: str, generations: list[str], partition: list[int]):
    rm, tokenizer = rm_and_tokenizer()
    convs = [
        [
            {"content": prompt, "role": "user"},
            {"content": generation, "role": "assistant"},
        ]
        for generation in generations
    ]
    batch = tokenizer.apply_chat_template(
        convs,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(rm.device)
    with torch.no_grad():
        raw_rewards = rm(**batch).logits[:, 0].tolist()

    scores = [transform_raw_reward(r) for r in raw_rewards]
    generation_scores = []
    partition_scores = []

    for s, p in zip(scores, partition, strict=False):
        if p == len(partition_scores):
            generation_scores.append(s)
            partition_scores.append(s)
        else:
            generation_scores.append(0)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores


async def process_instances(instances, output_file, patience, reference_prompt):
    if os.path.exists(output_file):
        try:
            existing_output = load_dataset("json", data_files=output_file, split="train")
            if not set(instances["id"]) - set(existing_output["id"]):
                print("All prompts are scored. Skipping.")
                return
        except datasets.exceptions.DatasetGenerationError:
            pass

    async with aio_open(output_file, "w", buffering=1) as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                generation_scores, partition_scores = await score_partition_rm(
                    reference_prompt,
                    instance["generations"],
                    instance["partition"],
                )
                utility = np.average(
                    generation_scores,
                    weights=patience ** np.arange(len(instance["generations"])),
                )
                return {
                    **instance,
                    "generation_scores": generation_scores,
                    "partition_scores": partition_scores,
                    "utility": utility,
                }

        tasks = [process_single_instance(instance) for instance in instances]
        for result in tqdm(await asyncio.gather(*tasks), total=len(instances)):
            await f.write(json.dumps(result) + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Score partitioned novelty-task generations with a reward model."
    )
    p.add_argument("--task", required=True, choices=list(TASK_CONFIG.keys()))
    p.add_argument("--patience", type=float, default=0.8,
                   help="Discount factor for cumulative utility.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--data-root", default="interaction/conversation_data")
    return p.parse_args()


async def main():
    args = parse_args()
    task_cfg = TASK_CONFIG[args.task]
    folder_idx = task_cfg["folder_idx"]
    reference_prompt = task_cfg["reference_prompt"]

    data_root = Path(args.data_root)
    base = data_root / f"novelty{folder_idx}" / "analysis"
    eval_dir = base / "partition" / f"temp{args.temperature}" / "1"
    out_dir = base / "score" / f"temp{args.temperature}" / "1"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = set(os.listdir(out_dir))
    filenames = [f for f in os.listdir(eval_dir) if f not in existing]

    for filename in filenames:
        instances = load_dataset(
            "json",
            data_files=str(eval_dir / filename),
            split="train",
        )
        await process_instances(instances, str(out_dir / filename), args.patience, reference_prompt)


if __name__ == "__main__":
    asyncio.run(main())
