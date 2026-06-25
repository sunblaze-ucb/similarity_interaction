import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, TextIO

import torch
import torch._dynamo as dynamo
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from tqdm import tqdm

from models.model import Model
import models.model_config

load_dotenv(find_dotenv())

ALPHABET = [chr(i) for i in range(ord("a"), ord("z") + 1)]


class OpenAIChat:
    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str | None = None, **defaults):
        self.model = model
        self.system_msg = {"role": "system", "content": system_prompt} if system_prompt else None
        self.defaults = defaults
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def __call__(self, messages: Iterable[Dict[str, str]], **overrides) -> str:
        if self.system_msg:
            messages = [self.system_msg, *messages]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=0,
            **{**self.defaults, **overrides},
        )
        return response.choices[0].message.content.strip()


def extract_word(response: str) -> str:
    gpt = OpenAIChat()
    prompt = f"In the following sentence, which word did they provide? Reply only with the word.\n\n{response}"
    return gpt([{"role": "user", "content": prompt}])


def converse(history: List[Dict[str, str]], model: Model, prompt: str,
             log_file: TextIO, round_idx: int, speaker: str,
             score: float, guess: bool, word: str = "") -> tuple[str, float]:
    history.append({"role": "user", "content": prompt})
    reply: str = model.generate_text([history])[0]
    history.append({"role": "assistant", "content": reply})

    reply = extract_word(reply)
    reply = re.sub(r"[^A-Za-z0-9 ]+", "", reply)

    if guess and word.lower() == reply.lower():
        score += 1

    log_file.write(json.dumps({
        "round": round_idx,
        "speaker": speaker,
        "total": score,
        "input": prompt,
        "text": reply,
    }, ensure_ascii=False) + "\n")
    log_file.flush()
    return reply, score


def cleanup(model1=None, model2=None):
    if model1 is not None:
        del model1
    if model2 is not None:
        del model2
    gc.collect()
    torch.cuda.empty_cache()
    dynamo.reset()
    torch.cuda.ipc_collect()


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the word-guessing game across model pairs (iterates the alphabet)."
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--repeats", type=int, nargs=2, metavar=("START", "END"), default=[1, 4],
                   help="Repeat folder range (inclusive).")
    p.add_argument("--output-root", default="conversation_data")
    return p.parse_args()


def play_game(model1, model2, model_id1, model_id2, outfile):
    score = 0.0
    with open(outfile, "w", encoding="utf-8") as f:
        for r in tqdm(range(len(ALPHABET))):
            letter = ALPHABET[r]
            history1, history2 = [], []

            prompt1 = f"Please think of a word that starts with '{letter}.' Please provide that word."
            target_word, score = converse(history1, model1, prompt1, f, r, model_id1, score, False)

            prompt2 = (
                f"Please provide a one-word hint so that others can guess your word: {target_word}. "
                "The hint word should be different from your word. Please provide the hint word."
            )
            hint, score = converse(history1, model1, prompt2, f, r, model_id1, score, False)

            prompt3 = (
                f'You must guess a word that starts with "{letter}." Here is a hint: "{hint}." '
                f'Respond with the single word starting with "{letter}" you think is being defined.'
            )
            _, score = converse(history2, model2, prompt3, f, r, model_id2, score, True, target_word)


def main():
    args = parse_args()
    model_list = models.model_config.MODEL_LIST
    start, end = args.repeats

    for repeat in range(start, end + 1):
        out_dir = Path(args.output_root) / "word-guessing" / f"temp{args.temperature}/{repeat}"
        out_dir.mkdir(parents=True, exist_ok=True)
        existing_files = os.listdir(out_dir)

        for model_id1 in model_list:
            for model_id2 in model_list:
                print(f"{model_id1}, {model_id2} conversation starts!")
                model_name1 = model_id1[model_id1.index("/") + 1:]
                model_name2 = model_id2[model_id2.index("/") + 1:]
                outfile = out_dir / f"{model_name1}_{model_name2}.jsonl"
                if f"{model_name1}_{model_name2}.jsonl" in existing_files:
                    continue

                model1 = model2 = None
                try:
                    model1 = Model(model_id1, args.temperature)
                    model2 = Model(model_id2, args.temperature)
                    existing_files.append(f"{model_name1}_{model_name2}.jsonl")
                    play_game(model1, model2, model_id1, model_id2, outfile)
                    cleanup(model1, model2)
                except Exception as e:
                    print(e)
                    cleanup(model1, model2)
                    if outfile.exists():
                        outfile.unlink()
                    continue


if __name__ == "__main__":
    main()
