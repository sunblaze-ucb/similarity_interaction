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
             log_file: TextIO, round_idx: int, speaker: str, total: float) -> str:
    history.append({"role": "user", "content": prompt})
    reply: str = model.generate_text([history])[0]
    if "<|assistant|>" in reply:
        reply = reply[:reply.index("<|assistant|>")]
    history.append({"role": "assistant", "content": reply})
    log_file.write(json.dumps({
        "round": round_idx,
        "speaker": speaker,
        "total": total,
        "input": prompt,
        "text": reply,
    }, ensure_ascii=False) + "\n")
    log_file.flush()
    return reply


def contribution(response: str, amount: float) -> float:
    gpt = OpenAIChat()
    prompt = (
        f"In the following sentences, how much of ${amount} do they decide to contribute? "
        f"Reply only with the number, between 0 and {amount}. Here are the sentences:\n\n"
        f'"{response}"'
    )
    return float(gpt([{"role": "user", "content": prompt}]).replace("$", ""))


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
        description="Run the public-good game across model pairs and log conversations."
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--repeats", type=int, nargs=2, metavar=("START", "END"), default=[1, 4],
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


def feedback_msg(other_response: str, public_pot: float, my_contrib: float) -> str:
    msg = (
        f'The contributions have now been collected! The other player decided its contribution as follows:\n'
        f'"{other_response}"\n\n'
        f'Therefore, after applying the 1.3x multiplication, the total value in the public pot is '
        f'${public_pot}. Therefore, you receive half of this amount, which is ${public_pot/2}. '
    )
    if public_pot / 2 > my_contrib:
        msg += f"As a result, you earn ${public_pot/2 - my_contrib}."
    elif public_pot / 2 == my_contrib:
        msg += "As a result, your net profit is 0."
    else:
        msg += f"As a result, you lose ${my_contrib - public_pot/2}."
    return msg


def play_game(model1, model2, model_id1, model_id2, args, outfile):
    duplicate_model = model2 is None

    with open(outfile, "w", encoding="utf-8") as f:
        history1, history2 = [], []
        game_intro = (
            "You will play a game with another one. Here's how it works:\n\n"
            "1. You begin with $100.\n"
            "2. Decide how much of your $100 you wish to contribute to a public pot. You can contribute "
            "any amount between $0 and $100.\n"
            "3. After both players have made their contributions, the total amount in the public pot "
            "will be multiplied by 1.3.\n"
            "4. The multiplied amount in the public pot will then be divided equally between you and the "
            "other player, regardless of how much each of you contributed.\n"
            "5. After the redistribution, you will keep whatever amount you receive.\n\n"
            "Now, please decide how much you'll contribute to the public pot out of $100. You should "
            "first say a reason for your decision and then your contribution amount.\n"
        )

        m1_resp = converse(history1, model1, game_intro, f, 0, model_id1, 100)
        if not duplicate_model:
            m2_resp = converse(history2, model2, game_intro, f, 0, model_id2, 100)
        else:
            m2_resp = m1_resp
            log_solo(f, 0, model_id2, 100, game_intro, m2_resp)

        m1_contrib = contribution(m1_resp, 100)
        m2_contrib = m1_contrib if duplicate_model else contribution(m2_resp, 100)

        public_pot = 1.3 * (m1_contrib + m2_contrib)
        game_end1 = feedback_msg(m2_resp, public_pot, m1_contrib)
        game_end2 = feedback_msg(m1_resp, public_pot, m2_contrib)

        total1 = 100 + public_pot / 2 - m1_contrib
        total2 = 100 + public_pot / 2 - m2_contrib

        m1_post = converse(history1, model1, game_end1, f, 1, model_id1, total1)
        if not duplicate_model:
            converse(history2, model2, game_end2, f, 1, model_id2, total2)
        else:
            log_solo(f, 1, model_id2, total2, game_end2, m1_post)

        for r in tqdm(range(1, args.rounds)):
            game_intro1 = (
                f"Now you have ${total1}. The same game repeats. Please decide how much you'll contribute "
                f"to the public pot out of ${total1}. You should first say a reason for your decision and "
                "then your contribution amount."
            )
            game_intro2 = (
                f"Now you have ${total2}. The same game repeats. Please decide how much you'll contribute "
                f"to the public pot out of ${total2}. You should first say a reason for your decision and "
                "then your contribution amount."
            )
            m1_resp = converse(history1, model1, game_intro1, f, r * 2, model_id1, total1)
            if not duplicate_model:
                m2_resp = converse(history2, model2, game_intro2, f, r * 2, model_id2, total2)
            else:
                m2_resp = m1_resp
                log_solo(f, r * 2, model_id2, total2, game_intro2, m2_resp)

            m1_contrib = contribution(m1_resp, total1)
            m2_contrib = m1_contrib if duplicate_model else contribution(m2_resp, total2)

            public_pot = 1.3 * (m1_contrib + m2_contrib)
            game_end1 = feedback_msg(m2_resp, public_pot, m1_contrib)
            game_end2 = feedback_msg(m1_resp, public_pot, m2_contrib)

            total1 = total1 + public_pot / 2 - m1_contrib
            total2 = total2 + public_pot / 2 - m2_contrib

            m1_post = converse(history1, model1, game_end1, f, r * 2 + 1, model_id1, total1)
            if not duplicate_model:
                converse(history2, model2, game_end2, f, r * 2 + 1, model_id2, total2)
            else:
                log_solo(f, r * 2 + 1, model_id2, total2, game_end2, m1_post)


def main():
    args = parse_args()
    model_list = models.model_config.MODEL_LIST
    start, end = args.repeats

    for repeat in range(start, end + 1):
        out_dir = Path(args.output_root) / "public-good" / f"temp{args.temperature}/{repeat}"
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
