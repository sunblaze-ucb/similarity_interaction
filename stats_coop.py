import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from calculate_similarity import calculate_similarity

DATASETS = ["wikitext_test_1000", "gsm8k", "math", "truthfulQA"]
GAMES = ["word-guessing", "public-good", "divide", "kbc"]
POOL_CHOICES = ["global", "max_aligned"]
LAYER_REGION_CHOICES = ["all", "early", "mid", "end"]


def metric_type_dirname(kernel_metric: str, unbiased: bool) -> str:
    kernel = {"ip": "l", "rbf": "r"}[kernel_metric]
    bias = "u" if unbiased else "b"
    return f"cka_{kernel}_{bias}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Mixed-effects analysis of representational similarity vs. cooperation game outcome."
    )
    p.add_argument("--dataset", default="wikitext_test_1000", choices=DATASETS)
    p.add_argument("--kernel-metric", default="ip", choices=["ip", "rbf"])
    p.add_argument("--unbiased", action="store_true",
                   help="Use the unbiased HSIC estimator (default: biased).")
    p.add_argument("--games", nargs="+", default=GAMES, choices=GAMES)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--pool", default="global", choices=POOL_CHOICES,
                   help="How to pool the layerwise CKA matrix to a scalar.")
    p.add_argument("--layer-region", default="all", choices=LAYER_REGION_CHOICES,
                   help="Restrict pooling to a layer block (thirds).")
    p.add_argument("--output-dir", default="results/coop")
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def compute_score(mat: np.ndarray, pool: str, layer_region: str) -> float:
    if layer_region != "all":
        nrows, ncols = mat.shape
        row1, row2 = nrows // 3, 2 * nrows // 3
        col1, col2 = ncols // 3, 2 * ncols // 3
        if layer_region == "early":
            mat = mat[:row1, :col1]
        elif layer_region == "mid":
            mat = mat[row1:row2, col1:col2]
        elif layer_region == "end":
            mat = mat[row2:, col2:]

    if pool == "global":
        return mat.mean()
    if pool == "max_aligned":
        return (mat.max(axis=1).mean() + mat.max(axis=0).mean()) / 2
    raise ValueError(f"Unknown pool: {pool}")


def fit_summary(mfit, terms: dict) -> dict:
    ci = mfit.conf_int()
    out = {}
    for label, term in terms.items():
        if term not in mfit.params.index:
            continue
        out[label] = {
            "coef": float(mfit.params[term]),
            "se": float(mfit.bse[term]),
            "p": float(mfit.pvalues[term]),
            "ci": [float(ci.loc[term][0]), float(ci.loc[term][1])],
        }
    return out


country_dict = {
    "Llama-3.2-3B-Instruct": "usa",
    "Llama-3.2-11B-Vision-Instruct": "usa",
    "Llama-3.3-70B-Instruct": "usa",
    "gemma-3-1b-it": "usa",
    "gemma-3-4b-it": "usa",
    "gemma-3-12b-it": "usa",
    "gemma-3-27b-it": "usa",
    "Qwen2.5-3B-Instruct": "china",
    "Qwen2.5-7B-Instruct": "china",
    "Qwen2.5-14B-Instruct": "china",
    "Qwen2.5-72B-Instruct": "china",
    "Falcon3-3B-Instruct": "uae",
    "Falcon3-7B-Instruct": "uae",
    "Falcon3-10B-Instruct": "uae",
    "Phi-3.5-mini-instruct": "usa",
    "Phi-3-medium-128k-instruct": "usa",
    "Phi-4-mini-instruct": "usa",
    "phi-4": "usa",
    "Mistral-Nemo-Instruct-2407": "france",
    "Ministral-8B-Instruct-2410": "france",
    "gpt-oss-20b": "usa",
    "OLMo-2-0425-1B-Instruct": "usa",
    "OLMo-2-1124-13B-Instruct": "usa",
}

tokenizer_dict = {
    "Llama-3.2-3B-Instruct": "tiktoken",
    "Llama-3.2-11B-Vision-Instruct": "tiktoken",
    "Llama-3.3-70B-Instruct": "tiktoken",
    "gemma-3-1b-it": "SentencePiece",
    "gemma-3-4b-it": "SentencePiece",
    "gemma-3-12b-it": "SentencePiece",
    "gemma-3-27b-it": "SentencePiece",
    "Qwen2.5-3B-Instruct": "BPE",
    "Qwen2.5-7B-Instruct": "BPE",
    "Qwen2.5-14B-Instruct": "BPE",
    "Qwen2.5-72B-Instruct": "BPE",
    "Falcon3-3B-Instruct": "BPE",
    "Falcon3-7B-Instruct": "BPE",
    "Falcon3-10B-Instruct": "BPE",
    "Phi-3.5-mini-instruct": "SentencePiece",
    "Phi-3-medium-128k-instruct": "SentencePiece",
    "Phi-4-mini-instruct": "tiktoken",
    "phi-4": "tiktoken",
    "Mistral-Nemo-Instruct-2407": "tekken",
    "Ministral-8B-Instruct-2410": "tekken",
    "gpt-oss-20b": "o200k_harmony",
    "OLMo-2-0425-1B-Instruct": "cl100k",
    "OLMo-2-1124-13B-Instruct": "cl100k",
}

size_dict = {
    "Llama-3.2-3B-Instruct": math.log10(3.21) + 9,
    "Llama-3.2-11B-Vision-Instruct": math.log10(10.6) + 9,
    "Llama-3.3-70B-Instruct": math.log10(70.6) + 9,
    "gemma-3-1b-it": math.log10(1.0) + 9,
    "gemma-3-4b-it": math.log10(4.0) + 9,
    "gemma-3-12b-it": math.log10(12.2) + 9,
    "gemma-3-27b-it": math.log10(27.0) + 9,
    "Qwen2.5-3B-Instruct": math.log10(3.09) + 9,
    "Qwen2.5-7B-Instruct": math.log10(7.61) + 9,
    "Qwen2.5-14B-Instruct": math.log10(14.7) + 9,
    "Qwen2.5-72B-Instruct": math.log10(72.7) + 9,
    "Falcon3-3B-Instruct": math.log10(3.23) + 9,
    "Falcon3-7B-Instruct": math.log10(7.46) + 9,
    "Falcon3-10B-Instruct": math.log10(10.3) + 9,
    "Phi-3.5-mini-instruct": math.log10(3.8) + 9,
    "Phi-3-medium-128k-instruct": math.log10(14) + 9,
    "Phi-4-mini-instruct": math.log10(3.8) + 9,
    "phi-4": math.log10(14.7) + 9,
    "Mistral-Nemo-Instruct-2407": math.log10(12.2) + 9,
    "Ministral-8B-Instruct-2410": math.log10(8.02) + 9,
    "gpt-oss-20b": math.log10(21.5) + 9,
    "OLMo-2-0425-1B-Instruct": math.log10(1.48) + 9,
    "OLMo-2-1124-13B-Instruct": math.log10(13.7) + 9,
}


def load_performance_dict():
    performance_dict = {}
    with open("mmlu_all_results.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            performance_dict[obj["model"][obj["model"].index("/") + 1:]] = obj["accuracy"]
    return performance_dict


def calculate_entropy(lines):
    conversation = ""
    for line in lines:
        obj = json.loads(line)
        conversation += obj["text"] + " "

    words = re.findall(r"\b\w+(?:'\w+)?\b", conversation.lower())
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    probs = [count / total_words for count in word_counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def collect_data(games, dataset, metric_type, temp_set, pool, layer_region, refresh):
    out_dir = Path(f"rep_sim/{dataset}")
    out_dir2 = Path(f"rep_sim/{dataset}/{metric_type}")

    sim, y_value_dict, output_similarity = {}, {}, {}
    family, identical, tokenizer, country = {}, {}, {}, {}
    rep_sim_files = os.listdir(out_dir2)

    for game in games:
        game_temp_dir = f"interaction/conversation_data/{game}/temp{temp_set}"
        scenarios = [
            f"{game}/temp{temp_set}/{i}"
            for i in range(1, len(os.listdir(game_temp_dir)) + 1)
        ]
        for scenario in scenarios:
            rep = int(scenario.split("/")[-1])
            files = os.listdir(f"interaction/conversation_data/{scenario}")
            for f_name in files:
                with open(f"interaction/conversation_data/{scenario}/{f_name}", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) == 0:
                    continue

                stem = f_name[: f_name.index(".jsonl")]
                model_name1, model_name2 = stem.split("_")

                model_id1 = model_id2 = None
                for i, line in enumerate(lines):
                    if i == 0:
                        model_id1 = json.loads(line)["speaker"]
                    elif i == len(lines) - 1:
                        model_id2 = json.loads(line)["speaker"]

                dataname = f"{stem}_{temp_set}_{rep}_{game}"

                country[dataname] = int(country_dict[model_name1] == country_dict[model_name2])
                family[dataname] = int(
                    model_id1[: model_id1.index("/")] == model_id2[: model_id2.index("/")]
                )
                identical[dataname] = int(model_id1 == model_id2)
                tokenizer[dataname] = int(tokenizer_dict[model_name1] == tokenizer_dict[model_name2])

                csv_ab = f"{model_name1}_{model_name2}.csv"
                csv_ba = f"{model_name2}_{model_name1}.csv"
                if csv_ab in rep_sim_files and not refresh:
                    mat = pd.read_csv(out_dir2 / csv_ab, header=None).to_numpy()
                elif csv_ba in rep_sim_files and not refresh:
                    mat = pd.read_csv(out_dir2 / csv_ba, header=None).to_numpy()
                else:
                    print(f"{stem}.csv", model_id1, model_id2)
                    try:
                        mat = calculate_similarity(
                            dataset=dataset,
                            model_id1=model_id1,
                            model_id2=model_id2,
                            metric="cka",
                            metric_parameter=None,
                            conversation_history1=[],
                            conversation_history2=[],
                            figure_name=f"{model_name1}_{model_name2}",
                            output_dir=out_dir,
                            model_tensor_filename1=model_name1,
                            model_tensor_filename2=model_name2,
                        )
                        np.savetxt(out_dir2 / csv_ab, mat, delimiter=",", fmt="%.6f")
                        rep_sim_files = os.listdir(out_dir2)
                    except Exception as e:
                        print(e)
                        continue

                sim[dataname] = compute_score(mat, pool, layer_region)

                if "word-guessing" in scenario and len(lines) == 78:
                    y_value_dict[dataname] = json.loads(lines[-1])["total"]
                elif "kbc" in scenario and len(lines) == 20:
                    y_value_dict[dataname] = [
                        json.loads(lines[-2])["score"],
                        json.loads(lines[-1])["score"],
                    ]
                    output_similarity[dataname] = abs(
                        json.loads(lines[-2])["score"] - json.loads(lines[-1])["score"]
                    )
                elif len(lines) == 20:
                    y_value_dict[dataname] = [
                        json.loads(lines[-2])["total"],
                        json.loads(lines[-1])["total"],
                    ]
                    output_similarity[dataname] = abs(
                        json.loads(lines[-2])["total"] - json.loads(lines[-1])["total"]
                    )
                else:
                    print(scenario, f_name, len(lines))

    return sim, y_value_dict, output_similarity, family, identical, tokenizer, country


def build_dataframe(sim, y_value_dict, output_similarity, family, identical,
                    tokenizer, country, performance_dict):
    common_keys = set(sim) & set(y_value_dict)
    rows = []

    for k in common_keys:
        parts = k.split("_")
        model_i, model_j = parts[0], parts[1]
        game = parts[-1]
        size_i, size_j = size_dict[model_i], size_dict[model_j]
        base = {
            "sim": float(sim[k]),
            "family": family[k],
            "identical": identical[k],
            "size_diff": abs(size_i - size_j),
            "mean_size": (size_i + size_j) / 2,
            "tokenizer": tokenizer[k],
            "game": game,
            "country": country[k],
            "performance_diff": abs(performance_dict[model_i] - performance_dict[model_j]),
        }

        if game == "word-guessing":
            row = dict(base)
            row.update({
                "model_i": model_i,
                "model_j": model_j,
                "model_i_size": size_i,
                "model_j_size": size_j,
                "y": float(y_value_dict[k]),
            })
            rows.append(row)
        else:
            for mi, mj, y_val in [
                (model_i, model_j, y_value_dict[k][0]),
                (model_j, model_i, y_value_dict[k][1]),
            ]:
                row = dict(base)
                row.update({
                    "model_i": mi,
                    "model_j": mj,
                    "model_i_size": size_dict[mi],
                    "model_j_size": size_dict[mj],
                    "y": float(y_val),
                    "output_similarity": output_similarity[k],
                })
                rows.append(row)

    df = pd.DataFrame(rows)
    for c in ("model_i", "model_j", "game", "country", "family", "tokenizer", "identical"):
        df[c] = df[c].astype("category")
    df["grp"] = 1
    df["y_mean_g"] = df.groupby("game")["y"].transform("mean")
    df["y_std_g"] = df.groupby("game")["y"].transform("std")
    df["z_y"] = (df["y"] - df["y_mean_g"]) / df["y_std_g"]
    df["z_sim"] = (df["sim"] - df["sim"].min()) / (df["sim"].max() - df["sim"].min())
    df["z_size_diff"] = (df["size_diff"] - df["size_diff"].min()) / (
        df["size_diff"].max() - df["size_diff"].min()
    )
    df["z_performance_diff"] = (df["performance_diff"] - df["performance_diff"].min()) / (
        df["performance_diff"].max() - df["performance_diff"].min()
    )
    return df


def run_analyses(df, games, dataset, metric_type, output_dir):
    vc = {"m_i": "0 + C(model_i)", "m_j": "0 + C(model_j)"}

    def fit(formula):
        return smf.mixedlm(formula, df, groups="grp", vc_formula=vc, re_formula="0").fit(
            method="lbfgs"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    def save(name, summary):
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(summary, f, indent=4)

    single_game = len(games) == 1
    only_game = games[0] if single_game else None

    if single_game:
        mfit = fit("y ~ sim")
        print("\nMixed-effects: y ~ sim\n")
        print(mfit.summary())
        save("sim_only", fit_summary(mfit, {"similarity": "sim"}))

        mfit = fit("y ~ sim + performance_diff")
        print("\nMixed-effects: y ~ sim + performance_diff\n")
        print(mfit.summary())
        save("with_performance", fit_summary(mfit, {
            "similarity": "sim",
            "performance_diff": "performance_diff",
        }))

        if only_game != "word-guessing":
            mfit = fit("y ~ output_similarity")
            print("\nMixed-effects: y ~ output_similarity\n")
            print(mfit.summary())
            save("output_sim_only", fit_summary(mfit, {
                "output_similarity": "output_similarity",
            }))

            mfit = fit("y ~ sim + output_similarity")
            print("\nMixed-effects: y ~ sim + output_similarity\n")
            print(mfit.summary())
            save("with_output_sim", fit_summary(mfit, {
                "similarity": "sim",
                "output_similarity": "output_similarity",
            }))

    mfit = fit("y ~ z_sim + z_performance_diff + game")
    print("\nMixed-effects: y ~ z_sim + z_performance_diff + game\n")
    print(mfit.summary())
    save("with_performance_z", fit_summary(mfit, {
        "similarity": "z_sim",
        "performance_diff": "z_performance_diff",
    }))

    mfit = fit("z_y ~ z_sim + game")
    print("\nMixed-effects: z_y ~ z_sim + game\n")
    print(mfit.summary())
    save("z_scored_sim", fit_summary(mfit, {"similarity": "z_sim"}))

    out = smf.mixedlm(
        "z_y ~ z_sim + z_size_diff + family + game + tokenizer + identical",
        df,
        groups="grp",
        vc_formula=vc,
    ).fit(method="lbfgs")
    print("\nMixed-effects: z_y ~ z_sim + z_size_diff + family + game + tokenizer + identical\n")
    print(out.summary())
    save("all_factors", fit_summary(out, {
        "similarity": "z_sim",
        "size_diff": "z_size_diff",
        "family": "family[T.1]",
        "tokenizer": "tokenizer[T.1]",
        "identical": "identical[T.1]",
    }))


def main():
    args = parse_args()
    metric_type = metric_type_dirname(args.kernel_metric, args.unbiased)
    games = sorted(args.games)
    games_str = "+".join(games)

    output_dir = (
        Path(args.output_dir)
        / metric_type
        / args.dataset
        / f"{games_str}_{args.pool}_{args.layer_region}"
    )

    performance_dict = load_performance_dict()
    sim, y_value_dict, output_similarity, family, identical, tokenizer, country = collect_data(
        games, args.dataset, metric_type, args.temp, args.pool, args.layer_region, args.refresh,
    )
    df = build_dataframe(
        sim, y_value_dict, output_similarity, family, identical, tokenizer, country,
        performance_dict,
    )
    run_analyses(df, games, args.dataset, metric_type, output_dir)


if __name__ == "__main__":
    main()
