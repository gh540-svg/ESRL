"""Run one independent Qwen3 domain-adaptation arm.

Unlike ``run_continual_baselines.py``, this runner never inherits weights from
another domain.  Every method/domain pair starts from the same frozen Qwen3
base, receives 2,000 prompt visits from exactly one training benchmark, and
writes one manifest for the in-domain/out-of-domain evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import run_continual_baselines as CL


BASELINE_METHODS = (
    "ttrl",
    "es_direct",
    "sft_mode",
    "echo",
    "evol_rl",
    "es_grpo_filter",
)
PUBLIC_METHODS = BASELINE_METHODS + (
    "es_grpo",
    "es_grpo_filter_n32",
    "es_direct_lora_vllm",
)
INTERNAL_METHOD = {
    "sft_mode": "mode_sft",
    "es_grpo_filter": "es_grpo_ff_n16",
    "es_grpo_filter_n32": "es_grpo_ff_n32",
}
DOMAINS = {
    "math": {
        "name": "math500",
        "train_size": 400,
        "eval_size": 100,
        "epochs": 5,
        "permute": False,
        "max_new": 512,
    },
    "knowledge": {
        "name": "mmlu",
        "train_size": 500,
        "eval_size": 200,
        "epochs": 4,
        "permute": True,
        "max_new": 256,
    },
    "medical": {
        "name": "medqa",
        "train_size": 500,
        "eval_size": 200,
        "epochs": 4,
        "permute": True,
        "max_new": 256,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=PUBLIC_METHODS)
    parser.add_argument("model", choices=tuple(CL.MODELS))
    parser.add_argument("domain", choices=tuple(DOMAINS))
    parser.add_argument(
        "--root", default=os.environ.get("IDOOD_ROOT", "out_qwen3_idood")
    )
    parser.add_argument(
        "--snap-every",
        type=int,
        default=int(os.environ.get("IDOOD_SNAP_EVERY", "500")),
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=float(os.environ.get("IDOOD_HOURS", "160")),
    )
    parser.add_argument(
        "--max-no-progress-retries",
        type=int,
        default=int(os.environ.get("IDOOD_MAX_NO_PROGRESS_RETRIES", "3")),
    )
    parser.add_argument("--kl-lambda", type=float, default=0.001)
    parser.add_argument("--anchor-block-size", type=int, default=10)
    parser.add_argument("--es-sigma", type=float, default=0.001)
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("IDOOD_SEED", "0")),
        help="sampling and data-split seed for an independent repetition",
    )
    parser.add_argument(
        "--run-tag", default=os.environ.get("IDOOD_RUN_TAG", ""),
        help="optional suffix used to keep reruns from overwriting prior arms",
    )
    return parser.parse_args()


def population_for(method):
    return 32 if method == "es_grpo_filter_n32" else 16


def arm_name(method, model_key, domain, run_tag=""):
    arm = (
        f"{method}_qwen3_{model_key}_train_{domain}_"
        f"n{population_for(method)}"
    )
    return f"{arm}_{run_tag}" if run_tag else arm


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if args.snap_every <= 0 or CL.VISITS % args.snap_every:
        raise SystemExit("--snap-every must be positive and divide 2000")
    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.run_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_tag):
        raise SystemExit(
            "--run-tag may contain only letters, numbers, dot, underscore, and dash"
        )

    public_method = args.method
    # The shared continual runner historically exposes the experiment seed as a
    # module constant. Override it before command construction so data splits,
    # rollout sampling, checkpoints, and the manifest all use this repetition's
    # requested seed.
    CL.SEED = args.seed
    method = INTERNAL_METHOD.get(public_method, public_method)
    stage = dict(DOMAINS[args.domain])
    population = population_for(public_method)
    arm = arm_name(public_method, args.model, args.domain, args.run_tag)
    output = Path(args.root) / arm / f"train_{stage['name']}"

    runner_args = argparse.Namespace(
        hours_per_stage=args.hours,
        max_no_progress_retries=args.max_no_progress_retries,
        snap_every=args.snap_every,
        kl_lambda=args.kl_lambda,
        anchor_block_size=args.anchor_block_size,
        es_sigma=args.es_sigma,
    )

    print("=" * 76)
    print(f"protocol=independent-domain-transfer arm={arm}")
    print(f"method={public_method} model={CL.MODELS[args.model]}")
    print(
        f"source_domain={args.domain} dataset={stage['name']} "
        f"unique_examples={stage['train_size']} visits={CL.VISITS}"
    )
    print("initial_weights=Qwen3 base (no cross-domain inheritance)")
    if public_method == "echo":
        print(
            "ECHO=fixed-budget N16; hybrid advantage + adaptive clipping; "
            "DeepConf tree branching disabled"
        )
    print("=" * 76, flush=True)

    final = CL.run_stage(
        runner_args,
        method,
        CL.MODELS[args.model],
        1,
        stage,
        output,
        inherited=None,
    )
    if final is None:
        raise SystemExit("training finished without a complete final snapshot")

    manifest = Path(f"manifest_idood_{arm}.json")
    payload = {
        "protocol": "qwen3_independent_domain_in_out_2000_v1",
        "seed": CL.SEED,
        "arm": arm,
        "method": public_method,
        "internal_method": method,
        "model_key": args.model,
        "model": CL.MODELS[args.model],
        "source_domain": args.domain,
        "training_dataset": stage["name"],
        "unique_training_examples": stage["train_size"],
        "epochs": stage["epochs"],
        "prompt_visits": CL.VISITS,
        "population": population,
        "snapshot_every": args.snap_every,
        "max_new_tokens": stage["max_new"],
        "choice_permutation": stage["permute"],
        "output_dir": str(output),
        "final_checkpoint": str(final),
        "es_grpo_anchor": {
            "teacher_refresh_prompts": args.anchor_block_size,
            "sigma": args.es_sigma,
            "kl_lambda": args.kl_lambda,
        } if public_method.startswith("es_grpo") else None,
        "frequency_filter": {
            "enabled": True,
            "fraction_retained": 0.5,
            "generated_rollouts": population,
            "retained_rollouts": population // 2,
            "ranking": "answer_frequency",
        } if public_method in (
            "es_grpo_filter", "es_grpo_filter_n32"
        ) else None,
        "echo": {
            "rollout_mode": "flat_fixed_budget",
            "deepconf_tree_branching": False,
            "alpha": 1.0,
            "beta": 1.0,
            "score_clamp": 2.0,
            "multiplicative_coefficient": 0.1,
            "tail_window": 16,
            "confidence_topk": 5,
            "clip_epsilon_min": 0.05,
            "clip_epsilon_max": 0.2,
            "clip_kappa": 4.0,
        } if public_method == "echo" else None,
        "split_policy": {
            "math": "MATH-500 seed-0: 400 adaptation, disjoint 100 holdout",
            "knowledge": "MMLU all/test seed-0: 500 adaptation, disjoint 200 holdout",
            "medical": "MedQA-USMLE train: 500 adaptation; native test evaluation",
        },
    }
    atomic_json(manifest, payload)
    print(f"[indomain-ood] complete; manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
