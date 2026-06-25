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
        "score": score,
        "input": prompt,
        "text": reply,
    }, ensure_ascii=False) + "\n")
    log_file.flush()
    return reply


def extract_number(response: str) -> float:
    gpt = OpenAIChat()
    prompt = (
        "In the following sentence, which number between 0 and 100 did they choose? "
        f'Reply with the number only. Here is the sentence:\n\n"{response}"'
    )
    return float(gpt([{"role": "user", "content": prompt}]))


def cleanup(model1=None, model2=None):
    for m in (model1, model2):
        if m is not None:
            try:
                m.to("cpu")
            except Exception:
                pass
            del m
    gc.collect()
    torch.cuda.empty_cache()
    dynamo.reset()
    torch.cuda.ipc_collect()


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Keynesian beauty contest (KBC) game across model pairs."
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--repeats", type=int, nargs=2, metavar=("START", "END"), default=[1, 4],
                   help="Repeat folder range (inclusive).")
    p.add_argument("--output-root", default="conversation_data")
    return p.parse_args()


def log_solo(f: TextIO, round_idx: int, speaker: str, score: float, prompt: str, text: str):
    f.write(json.dumps({
        "round": round_idx,
        "speaker": speaker,
        "score": score,
        "input": prompt,
        "text": text,
    }, ensure_ascii=False) + "\n")
    f.flush()


def feedback_msg(other_response: str, my_number: float, other_number: float,
                 average: float, my_score: float, other_score: float) -> str:
    return (
        f'Both players have chosen their numbers. The other player decided its number as follows:\n'
        f'"{other_response}"\n\n'
        f"Therefore, the two-thirds of the average of your number and the other player's number is "
        f"{average}. As a result, you receive a score of {100 - abs(my_number - average)} out of 100 "
        f"this round, and the other player receives {100 - abs(other_number - average)}. Your total "
        f"score is {my_score}, while the other player's total score is {other_score}."
    )


def play_game(model1, model2, model_id1, model_id2, args, outfile):
    score1 = score2 = 0.0
    duplicate_model = model2 is None

    with open(outfile, "w", encoding="utf-8") as f:
        history1, history2 = [], []
        game_intro = (
            "You will play a game with another one. Here's how it works:\n\n"
            "1. You must choose a number from 0 to 100.\n"
            "2. After both players choose, we take the average of the two numbers and then calculate "
            "two-thirds of that average. You will receive a score based on how close your chosen number "
            "is to this value. Specifically, your score is 100-|your guess-2/3*average|. In short, the "
            "closer your number is to 2/3 of the average, the higher your score.\n"
            "3. Your goal is to get the highest score.\n\n"
            "Now, please choose a number from 0 to 100. First give a sentence explaining your decision, "
            "and then state your number.\n"
        )

        m1_resp = converse(history1, model1, game_intro, f, 0, model_id1, score1)
        if not duplicate_model:
            m2_resp = converse(history2, model2, game_intro, f, 0, model_id2, score2)
        else:
            m2_resp = m1_resp
            log_solo(f, 0, model_id2, score2, game_intro, m2_resp)

        m1_num = extract_number(m1_resp)
        m2_num = m1_num if duplicate_model else extract_number(m2_resp)

        average = 1 / 3 * (m1_num + m2_num)
        score1 += 100 - abs(m1_num - average)
        score2 += 100 - abs(m2_num - average)

        game_end1 = feedback_msg(m2_resp, m1_num, m2_num, average, score1, score2)
        game_end2 = feedback_msg(m1_resp, m2_num, m1_num, average, score2, score1)

        m1_post = converse(history1, model1, game_end1, f, 1, model_id1, score1)
        if not duplicate_model:
            converse(history2, model2, game_end2, f, 1, model_id2, score2)
        else:
            log_solo(f, 1, model_id2, score2, game_end2, m1_post)

        for r in tqdm(range(1, args.rounds)):
            game_intro1 = (
                f"Now you have a score of {score1}. The same game repeats. Please decide a number between "
                "0 and 100. First give a sentence explaining your decision, and then state your number."
            )
            game_intro2 = (
                f"Now you have a score of {score2}. The same game repeats. Please decide a number between "
                "0 and 100. First give a sentence explaining your decision, and then state your number."
            )
            m1_resp = converse(history1, model1, game_intro1, f, r * 2, model_id1, score1)
            if not duplicate_model:
                m2_resp = converse(history2, model2, game_intro2, f, r * 2, model_id2, score2)
            else:
                m2_resp = m1_resp
                log_solo(f, r * 2, model_id2, score2, game_intro2, m2_resp)

            m1_num = extract_number(m1_resp)
            m2_num = m1_num if duplicate_model else extract_number(m2_resp)

            average = 1 / 3 * (m1_num + m2_num)
            score1 += 100 - abs(m1_num - average)
            score2 += 100 - abs(m2_num - average)

            game_end1 = feedback_msg(m2_resp, m1_num, m2_num, average, score1, score2)
            game_end2 = feedback_msg(m1_resp, m2_num, m1_num, average, score2, score1)

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
        out_dir = Path(args.output_root) / "kbc" / f"temp{args.temperature}/{repeat}"
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
