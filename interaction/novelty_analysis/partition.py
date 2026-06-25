import argparse
import asyncio
import functools
import json
import os
from pathlib import Path

import datasets
import sacrebleu
import torch
from aiofiles import open as aio_open
from datasets import Dataset, load_dataset
from evaluate import load
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CONCURRENT_REQUESTS = 1

TASK_FOLDER_IDX = {"story": 1, "biography": 2, "haiku": 3, "vacation": 4}

rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])
bertscorer = load("bertscore")


@functools.cache
def load_deberta_tokenizer_and_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yimingzhang/deberta-v3-large-generation-similarity"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


async def bleu(prompt: str, s1: str, s2: str):
    return (
        sacrebleu.corpus_bleu([s1], [[s2]]).score
        + sacrebleu.corpus_bleu([s2], [[s1]]).score
    ) / 200


async def rouge1(prompt: str, s1: str, s2: str):
    return rouge_scorer.score(s1, s2)["rouge1"].fmeasure


async def bertscore(prompt: str, s1: str, s2: str):
    return bertscorer.compute(
        predictions=[s1],
        references=[s2],
        model_type="microsoft/deberta-large",
    )["f1"][0]


@torch.inference_mode()
async def classifier_score(prompt: str, s1: str, s2: str):
    tokenizer, model = load_deberta_tokenizer_and_model()
    input_ids = [tokenizer.cls_token_id]
    for s in [s1, s2]:
        input_ids.extend(
            tokenizer.encode(
                s,
                truncation=True,
                max_length=128,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
        prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    return outputs["logits"].softmax(-1)[0, 1].cpu().item()


async def equivalence_check_unigram(prompt, response_0, response_1) -> bool:
    return await rouge1(prompt, response_0, response_1) > 0.458


async def equivalence_check_bertscore(prompt, response_0, response_1) -> bool:
    scores = await bertscore(prompt, response_0, response_1)
    return scores["f1"][0] > 0.719


def maybe_test_equality(response_0: str, response_1: str) -> bool | None:
    unigram_0 = response_0.strip().lower().split()
    unigram_1 = response_1.strip().lower().split()
    max_len = max(len(unigram_0), len(unigram_1))
    if max_len <= 5:
        return len(set(unigram_0) & set(unigram_1)) * 2 >= max_len
    return None


async def equivalence_check_classifier(prompt, response_0, response_1) -> bool:
    equality = maybe_test_equality(response_0, response_1)
    if equality is not None:
        return equality
    return await classifier_score(prompt, response_0, response_1) > 0.102


async def partition_responses(prompt, responses, equivalence_alg) -> list[int]:
    equivalence_classes = []
    partition = [-1] * len(responses)

    for i in range(len(responses)):
        if partition[i] >= 0:
            continue
        current_class = [responses[i]]
        partition[i] = len(equivalence_classes)
        for j in range(i + 1, len(responses)):
            if partition[j] == -1 and await equivalence_alg(
                prompt, current_class[0], responses[j]
            ):
                current_class.append(responses[j])
                partition[j] = len(equivalence_classes)
        equivalence_classes.append(current_class)

    assert all(p >= 0 for p in partition)
    return partition


EQUIVALENCE_ALGS = {
    "unigram": equivalence_check_unigram,
    "bertscore": equivalence_check_bertscore,
    "classifier": equivalence_check_classifier,
}


async def process_instances(instances, output_file, equivalence_alg):
    if os.path.exists(output_file):
        try:
            existing_output = load_dataset("json", data_files=output_file, split="train")
            if not set(instances["id"]) - set(existing_output["id"]):
                print("All prompts have been partitioned. Skipping.")
                return
        except datasets.exceptions.DatasetGenerationError:
            pass

    async with aio_open(output_file, "w", buffering=1) as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                partition = await partition_responses(
                    instance["prompt"],
                    instance["generations"],
                    equivalence_alg,
                )
                return {**instance, "partition": partition, "distinct": max(partition)}

        tasks = [process_single_instance(instance) for instance in instances]
        for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
            result = await task
            await f.write(json.dumps(result) + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Partition novelty-task generations into equivalence classes."
    )
    p.add_argument("--task", required=True, choices=list(TASK_FOLDER_IDX.keys()))
    p.add_argument("--alg", default="classifier", choices=EQUIVALENCE_ALGS.keys(),
                   help="Equivalence-testing method.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--data-root", default="interaction/conversation_data",
                   help="Root of conversation data folders.")
    return p.parse_args()


async def main():
    args = parse_args()
    folder_idx = TASK_FOLDER_IDX[args.task]
    equivalence_alg = EQUIVALENCE_ALGS[args.alg]

    data_root = Path(args.data_root)
    eval_dir = data_root / f"novelty{folder_idx}" / f"temp{args.temperature}" / "1"
    out_dir = data_root / f"novelty{folder_idx}" / "analysis" / "partition" / f"temp{args.temperature}" / "1"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = set(os.listdir(out_dir))
    filenames = [f for f in os.listdir(eval_dir) if f not in existing]

    for filename in filenames:
        data_file = eval_dir / filename
        with open(data_file, encoding="utf-8") as fh:
            lines = fh.readlines()
        if len(lines) < 40:
            continue

        current, current2 = {}, {}
        for i, line in enumerate(lines):
            obj = json.loads(line)
            if i == 0:
                current["prompt"] = obj["input"]
                current2["prompt"] = obj["input"]
                current["generations"] = []
                current2["generations"] = []
                current["speaker"] = obj["speaker"]
            elif i == 1:
                current2["speaker"] = obj["speaker"]

            if i % 4 == 2:
                current["generations"].append(obj["text"])
            elif i % 4 == 3:
                current2["generations"].append(obj["text"])

        instances = Dataset.from_list([current, current2])
        await process_instances(instances, str(out_dir / filename), equivalence_alg)


if __name__ == "__main__":
    asyncio.run(main())
