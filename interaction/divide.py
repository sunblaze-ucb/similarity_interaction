import argparse
import gc
import json
import os
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


def converse(history: List[Dict[str, str]], model: Model, prompt: str,
             log_file: TextIO, round_idx: int, speaker: str, score: float) -> str:
    history.append({"role": "user", "content": prompt})
    reply: str = model.generate_text([history])[0]
    history.append({"role": "assistant", "content": reply})
    log_file.write(json.dumps({
        "round": round_idx,
        "speaker": speaker,
        "total": score,
        "input": prompt,
        "text": reply,
    }, ensure_ascii=False) + "\n")
    log_file.flush()
    return reply


def extract_number(response: str) -> float:
    gpt = OpenAIChat()
    prompt = (
        "In the following sentence, how much of the $1 do they want? "
        "Reply with the number only. Please note that the number should be less than or equal to 1. "
        f'Here are the sentences:\n\n"{response}"'
    )
    demand = gpt([{"role": "user", "content": prompt}]).replace("$", "")
    return float(demand)


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
        description="Run the divide-the-dollar game across model pairs and log conversations."
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--repeats", type=int, nargs=2, metavar=("START", "END"), default=[1, 8],
                   help="Repeat folder range (inclusive).")
    p.add_argument("--output-root", default="conversation_data")
    return p.parse_args()


def log_solo(f: TextIO, round_idx: int, speaker: str, total: float, prompt: str, text: str):
    f.write(json.dumps({
        "round": round_idx,
        "speaker": speaker,
        "total": total,
        "input": prompt,
        "text": text,
    }, ensure_ascii=False) + "\n")
    f.flush()


def play_game(model1, model2, model_id1, model_id2, args, outfile):
    score1 = score2 = 0.0
    duplicate_model = model2 is None

    with open(outfile, "w", encoding="utf-8") as f:
        history1, history2 = [], []
        game_intro = (
            "You will play a game with another one. Here's how it works:\n\n"
            "1. There is $1 available. You must decide how much of this $1 you want. "
            "The other player will decide for themselves independently.\n"
            "2. If the sum of the two demands is less than or equal to $1, each player receives "
            "exactly the amount they demanded. If the sum exceeds $1, both players receive nothing.\n"
            "3. Your goal is to maximize the amount of money you receive.\n\n"
            "Now, please state your demand. First give a sentence explaining your decision, "
            "then state the amount you demand."
        )

        m1_resp = converse(history1, model1, game_intro, f, 0, model_id1, score1)
        if not duplicate_model:
            m2_resp = converse(history2, model2, game_intro, f, 0, model_id2, score2)
        else:
            m2_resp = m1_resp
            log_solo(f, 0, model_id2, score2, game_intro, m2_resp)

        m1_num = extract_number(m1_resp)
        m2_num = m1_num if duplicate_model else extract_number(m2_resp)

        if m1_num + m2_num <= 1:
            score1 += m1_num
            score2 += m2_num
        else:
            print(m1_num, m2_num)
            m1_num = m2_num = 0

        game_end1 = (
            "Both players have decided how much they want. The other player decided as follows:\n"
            f'"{m2_resp}"\n\nTherefore, you receive ${m1_num} this round. As a result, your total '
            f"asset value is {score1}. Please wait for the next round of the game!"
        )
        game_end2 = (
            "Both players have decided how much they want. The other player decided as follows:\n"
            f'"{m1_resp}"\n\nTherefore, you receive ${m2_num} this round. As a result, your total '
            f"asset value is {score2}. Please wait for the next round of the game!"
        )

        m1_post = converse(history1, model1, game_end1, f, 1, model_id1, score1)
        if not duplicate_model:
            converse(history2, model2, game_end2, f, 1, model_id2, score2)
        else:
            log_solo(f, 1, model_id2, score2, game_end2, m1_post)

        for r in tqdm(range(1, args.rounds)):
            again = (
                "The same game repeats. $1 is available again. Decide how much of that $1 you want. "
                "First give a sentence explaining your decision, then state the amount you demand."
            )
            m1_resp = converse(history1, model1, again, f, r * 2, model_id1, score1)
            if not duplicate_model:
                m2_resp = converse(history2, model2, again, f, r * 2, model_id2, score2)
            else:
                m2_resp = m1_resp
                log_solo(f, r * 2, model_id2, score2, again, m2_resp)

            m1_num = extract_number(m1_resp)
            m2_num = m1_num if duplicate_model else extract_number(m2_resp)

            if m1_num + m2_num <= 1:
                score1 += m1_num
                score2 += m2_num
            else:
                print(m1_num, m2_num)
                m1_num = m2_num = 0

            game_end1 = (
                f'Both players have decided how much they want. The other player decided as follows: "{m2_resp}"\n\n'
                f"Therefore, you receive ${m1_num} this round. As a result, your total asset value is {score1}. "
                "Please wait for the next round of the game!"
            )
            game_end2 = (
                f'Both players have decided how much they want. The other player decided as follows: "{m1_resp}"\n\n'
                f"Therefore, you receive ${m2_num} this round. As a result, your total asset value is {score2}. "
                "Please wait for the next round of the game!"
            )
            m1_post = converse(history1, model1, game_end1, f, r * 2 + 1, model_id1, score1)
            if not duplicate_model:
                converse(history2, model2, game_end2, f, r * 2 + 1, model_id2, score2)
            else:
                log_solo(f, r * 2 + 1, model_id2, score2, game_end2, m1_post)


def main():
    args = parse_args()
    model_list = models.model_config.MODEL_LIST
    start, end = args.repeats

    for repeat in range(start, end + 1):
        out_dir = Path(args.output_root) / "divide" / f"temp{args.temperature}/{repeat}"
        out_dir.mkdir(parents=True, exist_ok=True)
        existing_files = os.listdir(out_dir)

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
                    play_game(model1, model2, model_id1, model_id2, args, outfile)
                    cleanup(model1, model2)
                except Exception as e:
                    print(e)
                    cleanup(model1, model2)
                    if outfile.exists():
                        outfile.unlink()
                    continue


if __name__ == "__main__":
    main()
