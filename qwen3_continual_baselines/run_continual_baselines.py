"""Run one leakage-safe Qwen3 continual-learning baseline.

Stages are fixed to MATH-500 -> MMLU -> MedQA-USMLE.  Every stage receives
exactly 2,000 prompt visits.  One invocation runs one method/backbone pair so
Slurm can place each comparison arm on its own H200.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


MODELS = {
    "0p6b": "Qwen/Qwen3-0.6B-Base",
    "4b": "Qwen/Qwen3-4B-Base",
    "8b": "Qwen/Qwen3-8B-Base",
}
METHODS = (
    "es_direct", "es_direct_lora", "es_direct_lora_batch16",
    "es_direct_lora_vllm",
    "ttrl", "sft", "mode_sft", "echo", "evol_rl",
    "es_grpo", "es_grpo_p8_r2", "es_grpo_ff_p8_r2",
    "es_grpo_ff_n16", "es_grpo_ff_n32",
    "es_grpo_ff_n16_tok1024",
)
POPULATION = 16
SEED = 0
VISITS = 2000

STAGES = (
    {
        "name": "math500",
        "train_size": 400,
        "eval_size": 100,
        "epochs": 5,
        "permute": False,
        "max_new": 512,
    },
    {
        "name": "mmlu",
        "train_size": 500,
        "eval_size": 200,
        "epochs": 4,
        "permute": True,
        "max_new": 256,
    },
    {
        "name": "medqa",
        "train_size": 500,
        "eval_size": 200,
        "epochs": 4,
        "permute": True,
        "max_new": 256,
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--root", default=os.environ.get("CL_ROOT", "out_qwen3_cl"))
    parser.add_argument(
        "--snap-every",
        type=int,
        default=int(os.environ.get("CL_SNAP_EVERY", "500")),
    )
    parser.add_argument(
        "--hours-per-stage",
        type=float,
        default=float(os.environ.get("CL_HOURS_PER_STAGE", "55")),
    )
    parser.add_argument(
        "--max-no-progress-retries",
        type=int,
        default=int(os.environ.get("CL_MAX_NO_PROGRESS_RETRIES", "3")),
        help=("stop a stage after this many consecutive crashed subprocesses "
              "without a new checkpoint; prevents deterministic failures from "
              "holding a GPU for the entire retry budget"),
    )
    parser.add_argument(
        "--run-tag",
        default=os.environ.get("CL_RUN_TAG", ""),
        help="optional safe suffix used to isolate sweep outputs",
    )
    parser.add_argument(
        "--kl-lambda",
        type=float,
        default=float(os.environ.get("CL_ES_GRPO_KL_LAMBDA", "0.001")),
        help="ES-GRPO KL anchor strength against the frozen base model",
    )
    parser.add_argument(
        "--anchor-block-size",
        type=int,
        default=int(os.environ.get("CL_ES_GRPO_ANCHOR_BLOCK_SIZE", "10")),
        help="prompts between hard re-anchors of the ES teacher to the student",
    )
    parser.add_argument(
        "--es-sigma",
        type=float,
        default=float(os.environ.get("CL_ES_GRPO_SIGMA", "0.001")),
        help="standard deviation of ES perturbations around the teacher anchor",
    )
    return parser.parse_args()


def population_for(method):
    if method == "sft":
        return None
    if method in ("es_grpo_p8_r2", "es_grpo_ff_p8_r2"):
        return 8
    if method == "es_grpo_ff_n32":
        return 32
    return POPULATION


def rollouts_per_member_for(method):
    return 2 if method in ("es_grpo_p8_r2", "es_grpo_ff_p8_r2") else 1


def max_new_for(method, stage):
    if method == "es_grpo_ff_n16_tok1024":
        return 1024
    return stage["max_new"]


def arm_name(method, model_key):
    population = population_for(method)
    if method == "es_direct_lora_batch16":
        label = "es_direct_lora_batch16"
    elif method == "es_direct_lora_vllm":
        label = "es_direct_lora_vllm"
    elif method == "es_grpo_p8_r2":
        return f"es_grpo_p8_r2_qwen3_{model_key}"
    elif method == "es_grpo_ff_p8_r2":
        return f"es_grpo_ff_p8_r2_qwen3_{model_key}"
    elif method == "es_grpo_ff_n16_tok1024":
        label = "es_grpo_ff_tok1024"
    elif method.startswith("es_grpo_ff_"):
        label = "es_grpo_ff"
    else:
        label = method
    suffix = "" if population is None else f"_n{population}"
    return f"{label}_qwen3_{model_key}{suffix}"


def tagged_arm_name(method, model_key, run_tag=""):
    if run_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", run_tag):
        raise ValueError("--run-tag may contain only letters, digits, _, ., and -")
    base = arm_name(method, model_key)
    return f"{base}_{run_tag}" if run_tag else base


def snapshot_visit(path):
    digits = "".join(character for character in path.name if character.isdigit())
    return int(digits) if digits else -1


def complete_adapter(path):
    return (
        (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
        and (path / "adapter_model.safetensors").stat().st_size > 0
    )


def complete_full_model(path):
    if not (path / "config.json").is_file():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.is_file() and candidate.stat().st_size > 100_000_000:
            return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            shards = set(weight_map.values())
            return bool(shards) and all((path / shard).is_file() for shard in shards)
        except (OSError, KeyError, TypeError, ValueError):
            return False
    return False


def latest_snapshot(out, method):
    if not out.is_dir():
        return None
    complete = complete_full_model if method == "es_direct" else complete_adapter
    snapshots = sorted(
        (path for path in out.glob("snap[0-9]*") if path.is_dir()),
        key=snapshot_visit,
        reverse=True,
    )
    return next((path for path in snapshots if complete(path)), None)


def wait_for_cuda(max_wait=1800):
    waited, delay = 0, 15
    while waited <= max_wait:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; assert torch.cuda.is_available(); "
                "torch.zeros(1, device='cuda')",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        print(f"[preflight] CUDA unhealthy; retrying in {delay}s", flush=True)
        time.sleep(delay)
        waited += delay
        delay = min(delay * 2, 120)
    return False


def command_for(method, model, stage, out, source, current, snap_every,
                kl_lambda=0.001, anchor_block_size=10, es_sigma=0.001):
    shared = [
        "--model", model,
        "--dataset", stage["name"],
        "--train-size", str(stage["train_size"]),
        "--eval-size", str(stage["eval_size"]),
        "--epochs", str(stage["epochs"]),
        "--snap-every", str(snap_every),
        "--dtype", "bfloat16",
        "--seed", str(SEED),
        "--out", str(out),
    ]
    if stage["permute"]:
        shared.append("--permute-choices")
    if source is not None:
        shared.extend(["--init-adapter", str(source)])
    if current:
        shared.extend(["--start-visit", str(current)])

    if method == "sft":
        return [sys.executable, "-u", "train_sft_supervised.py"] + shared + [
            "--lr", "5e-5", "--lora-r", "16", "--max-length", "4096"
        ]

    method_args = [
        "--population", str(population_for(method)),
        "--rollouts-per-member", str(rollouts_per_member_for(method)),
        "--temp", "0.8",
        "--p-high", "0.95",
        "--max-new", str(max_new_for(method, stage)),
    ]
    if method == "mode_sft":
        executable = "train.py"
        method_args += [
            "--method", "sft_good",
            "--reward", "vote",
            "--lr", "5e-5",
            "--lora-r", "16",
        ]
    elif method == "es_direct_lora":
        executable = "train.py"
        method_args += [
            "--method", "es_direct_lora",
            "--reward", "vote",
            "--sigma", "0.001",
            "--alpha", "0.0005",
            "--lora-r", "16",
        ]
    elif method == "es_direct_lora_batch16":
        executable = "train.py"
        method_args += [
            "--method", "es_direct_lora_batch16",
            "--reward", "vote",
            "--sigma", "0.001",
            "--alpha", "0.0005",
            "--lora-r", "16",
            "--es-lora-batch-size", str(population_for(method)),
        ]
    elif method == "es_direct_lora_vllm":
        executable = "train_es_direct_lora_vllm.py"
        method_args += [
            "--sigma", "0.001",
            "--alpha", "0.0005",
            "--lora-r", "16",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "4096",
        ]
    elif method == "ttrl":
        executable = "train.py"
        method_args += [
            "--method", "grpo_real",
            "--reward", "vote",
            "--kl-lambda", "0.001",
            "--clip-eps", "0.2",
            "--block-size", "10",
            "--rollout-prompt-batch-size",
            os.environ.get("QWEN3_ROLLOUT_PROMPT_BATCH_SIZE", "1"),
            "--lora-r", "16",
        ]
    elif method == "echo":
        executable = "train.py"
        method_args += [
            "--method", "echo",
            "--reward", "vote",
            "--kl-lambda", "0.001",
            "--block-size", "10",
            "--rollout-prompt-batch-size",
            os.environ.get("QWEN3_ROLLOUT_PROMPT_BATCH_SIZE", "1"),
            "--lora-r", "16",
            "--echo-alpha", "1.0",
            "--echo-beta", "1.0",
            "--echo-s-max", "2.0",
            "--echo-mult-coeff", "0.1",
            "--echo-tail-window", "16",
            "--echo-confidence-topk", "5",
            "--echo-eps-min", "0.05",
            "--echo-eps-max", "0.2",
            "--echo-kappa", "4.0",
        ]
    elif method == "evol_rl":
        executable = "train_evol_rl.py"
        method_args += [
            "--kl-lambda", "0.001",
            "--clip-eps", "0.2",
            "--block-size", "10",
            "--rollout-prompt-batch-size",
            os.environ.get("QWEN3_ROLLOUT_PROMPT_BATCH_SIZE", "1"),
            "--lora-r", "16",
        ]
    elif method in ("es_grpo", "es_grpo_p8_r2"):
        executable = "train.py"
        method_args += [
            "--method", "es_grpo",
            "--kl-lambda", str(kl_lambda),
            "--block-size", str(anchor_block_size),
            "--sigma", str(es_sigma),
            "--lora-r", "16",
        ]
    elif method in (
        "es_grpo_ff_p8_r2", "es_grpo_ff_n16", "es_grpo_ff_n32",
        "es_grpo_ff_n16_tok1024"
    ):
        executable = "train.py"
        method_args += [
            "--method", "es_grpo",
            "--kl-lambda", str(kl_lambda),
            "--block-size", str(anchor_block_size),
            "--sigma", str(es_sigma),
            "--freq-filter", "0.5",
            "--lora-r", "16",
        ]
    elif method == "es_direct":
        executable = "train.py"
        method_args += [
            "--method", "es_direct",
            "--reward", "vote",
            "--sigma", "0.001",
            "--alpha", "0.0005",
        ]
    else:
        raise ValueError(f"unsupported method {method!r}")
    return [sys.executable, "-u", executable] + method_args + shared


def write_manifest(path, arm, method, model, chain, snap_every,
                   kl_lambda=0.001, anchor_block_size=10, es_sigma=0.001):
    payload = {
        "protocol": "math500_mmlu_medqa_2000",
        "seed": SEED,
        "population": population_for(method),
        "rollouts_per_population_member": rollouts_per_member_for(method),
        "total_rollouts_per_prompt": (
            None if population_for(method) is None else
            population_for(method) * rollouts_per_member_for(method)
        ),
        "simultaneous_rollouts": (
            population_for(method)
            if method in ("es_direct_lora_batch16", "es_direct_lora_vllm")
            else None
        ),
        "rollout_engine": (
            "vllm-0.11-v1-persistent-multilora"
            if method == "es_direct_lora_vllm" else "transformers"
        ),
        "frequency_filter": (
            0.5 if method in (
                "es_grpo_ff_p8_r2", "es_grpo_ff_n16", "es_grpo_ff_n32",
                "es_grpo_ff_n16_tok1024",
            ) else 0.0
        ),
        "anchor": {
            "es_teacher_center": "current_student_effective_weights",
            "teacher_reanchor_every_prompts": anchor_block_size,
            "es_perturbation_sigma": es_sigma,
            "kl_reference": "frozen_original_base_model",
            "kl_lambda": kl_lambda,
        },
        "max_new_tokens_by_stage": {
            stage["name"]: max_new_for(method, stage) for stage in STAGES
        },
        "visits_per_stage": VISITS,
        "snapshot_every": snap_every,
        "model": model,
        "method": method,
        "stage_order": [stage["name"] for stage in STAGES],
        "split_policy": {
            "math500": "seed-0 shuffle: first 100 eval, next 400 adaptation",
            "mmlu": "all/test seed-0 shuffle: first 200 eval, next 500 adaptation",
            "medqa": "native train seed-0 first 500 adaptation; native test first 200 eval",
        },
        "methods": {arm: chain},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage(args, method, model, stage_index, stage, out, inherited):
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours_per_stage * 3600
    backoff = 30
    no_progress_attempts = 0
    while time.time() < deadline:
        latest = latest_snapshot(out, method)
        current = snapshot_visit(latest) if latest else 0
        if current >= VISITS:
            return latest
        if not wait_for_cuda():
            continue
        source = latest or inherited
        command = command_for(
            method, model, stage, out, source, current, args.snap_every,
            args.kl_lambda, args.anchor_block_size, args.es_sigma,
        )
        print(
            f"\n[continual] s{stage_index}:{stage['name']} "
            f"{current}->{VISITS}; source={source or model}",
            flush=True,
        )
        result = subprocess.run(command)
        new_latest = latest_snapshot(out, method)
        after = snapshot_visit(new_latest) if new_latest else 0
        if after > current:
            print(f"[continual] progress {current}->{after} (rc={result.returncode})")
            backoff = 30
            no_progress_attempts = 0
        else:
            no_progress_attempts += 1
            print(
                f"[continual] no checkpoint progress (rc={result.returncode}); "
                f"attempt {no_progress_attempts}/{args.max_no_progress_retries}",
                flush=True,
            )
            if no_progress_attempts >= args.max_no_progress_retries:
                raise SystemExit(
                    f"stage {stage_index}:{stage['name']} failed "
                    f"{no_progress_attempts} consecutive times without producing "
                    "a checkpoint; inspect the Slurm .err file"
                )
            print(f"[continual] retrying in {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
    raise SystemExit(
        f"stage {stage_index}:{stage['name']} exceeded its "
        f"{args.hours_per_stage:g}h retry budget"
    )


def main():
    args = parse_args()
    if args.snap_every <= 0 or VISITS % args.snap_every:
        raise SystemExit("--snap-every must be positive and divide 2000")
    if args.kl_lambda < 0:
        raise SystemExit("--kl-lambda must be non-negative")
    if args.anchor_block_size <= 0:
        raise SystemExit("--anchor-block-size must be positive")
    if args.es_sigma < 0:
        raise SystemExit("--es-sigma must be non-negative")
    method, model = args.method, MODELS[args.model]
    arm = tagged_arm_name(method, args.model, args.run_tag)
    root = Path(args.root)
    manifest = Path(f"manifest_{arm}.json")

    print("=" * 76)
    print(f"arm={arm} model={model} method={method}")
    print("order=MATH-500 -> MMLU -> MedQA-USMLE")
    population = population_for(method)
    if population is None:
        print("visits=2000/stage population=n/a")
    else:
        rollouts_per_member = rollouts_per_member_for(method)
        print(f"visits=2000/stage population={population} "
              f"rollouts/member={rollouts_per_member} "
              f"total-rollouts/prompt={population * rollouts_per_member}")
    if method.startswith("es_grpo"):
        print(f"anchor: kl_lambda={args.kl_lambda:g} "
              f"teacher_refresh={args.anchor_block_size} prompts "
              f"sigma={args.es_sigma:g}")
    print("=" * 76, flush=True)

    inherited = None
    chain = []
    for index, stage in enumerate(STAGES, 1):
        out = root / arm / f"s{index}_{stage['name']}"
        final = run_stage(args, method, model, index, stage, out, inherited)
        chain.append(
            {
                "stage": index,
                "dataset": stage["name"],
                "train_size": stage["train_size"],
                "epochs": stage["epochs"],
                "max_new_tokens": max_new_for(method, stage),
                "dir": str(out),
                "final": str(final),
            }
        )
        write_manifest(
            manifest, arm, method, model, chain, args.snap_every,
            args.kl_lambda, args.anchor_block_size, args.es_sigma,
        )
        inherited = final
    print(f"[continual] complete; manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
