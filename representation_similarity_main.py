import argparse
import os
from pathlib import Path

import numpy as np

import interaction.models.model_config
from calculate_similarity import calculate_similarity

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

DATASETS = ["wikitext_test_1000", "gsm8k", "math", "truthfulQA"]


def metric_type_dirname(kernel_metric: str, unbiased: bool) -> str:
    kernel = {"ip": "l", "rbf": "r"}[kernel_metric]
    bias = "u" if unbiased else "b"
    return f"cka_{kernel}_{bias}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute pairwise layer-wise CKA similarity matrices for all model pairs."
    )
    p.add_argument("--dataset", default="truthfulQA", choices=DATASETS,
                   help="Probe dataset name (expects data/<dataset>.jsonl).")
    p.add_argument("--kernel-metric", default="ip", choices=["ip", "rbf"],
                   help="CKA kernel: 'ip' (linear) or 'rbf'.")
    p.add_argument("--unbiased", action="store_true",
                   help="Use the unbiased HSIC estimator (default: biased).")
    p.add_argument("--refresh", action="store_true",
                   help="Recompute pairs even if their CSV already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    metric_parameter = {"kernel_metric": args.kernel_metric, "unbiased": args.unbiased}
    metric_type = metric_type_dirname(args.kernel_metric, args.unbiased)

    model_list = interaction.models.model_config.MODEL_LIST

    out_dir = Path(f"rep_sim/{args.dataset}")
    out_dir2 = out_dir / metric_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir2.mkdir(parents=True, exist_ok=True)

    finish = set(os.listdir(out_dir2))

    for model_id1 in model_list:
        for model_id2 in model_list:
            model_name1 = model_id1[model_id1.index("/") + 1:]
            model_name2 = model_id2[model_id2.index("/") + 1:]
            f_name = f"{model_name1}_{model_name2}.csv"
            f_name_rev = f"{model_name2}_{model_name1}.csv"

            if not args.refresh and (f_name in finish or f_name_rev in finish):
                continue

            print(f"{f_name} starts!")
            sim_mat = calculate_similarity(
                dataset=args.dataset,
                model_id1=model_id1,
                model_id2=model_id2,
                metric="cka",
                metric_parameter=metric_parameter,
                conversation_history1=[],
                conversation_history2=[],
                figure_name=f"{model_name1}_{model_name2}",
                output_dir=out_dir,
                model_tensor_filename1=model_name1,
                model_tensor_filename2=model_name2,
                refresh=args.refresh,
            )
            np.savetxt(out_dir2 / f_name, sim_mat, delimiter=",", fmt="%.6f")

            finish.add(f_name)
            if model_name1 != model_name2:
                finish.add(f_name_rev)


if __name__ == "__main__":
    main()
