"""Evaluate one independent adaptation checkpoint on all nine benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import eval_continual_qwen3 as E
from run_continual_baselines import MODELS


TASK_DOMAIN = {
    "math500_holdout": "math",
    "amc23": "math",
    "aime25": "math",
    "mmlu_holdout": "knowledge",
    "arc_challenge": "knowledge",
    "gpqa_diamond": "knowledge",
    "medqa_test": "medical",
    "medmcqa_validation": "medical",
    "pubmedqa_labeled": "medical",
}
TASKS = tuple(TASK_DOMAIN)


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", action="store_true")
    source.add_argument("--manifest")
    parser.add_argument("--model-key", choices=tuple(MODELS), default="4b")
    parser.add_argument("--results", required=True)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_existing(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    except (OSError, ValueError):
        return []


def atomic_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if args.base:
        model_name = MODELS[args.model_key]
        arm = f"base_qwen3_{args.model_key}"
        method = "base"
        source_domain = "base"
        checkpoint = None
        visit = 0
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        model_name = manifest["model"]
        arm = manifest["arm"]
        method = manifest["method"]
        source_domain = manifest["source_domain"]
        checkpoint = manifest["final_checkpoint"]
        visit = int(manifest["prompt_visits"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = {
        task: E.load_core(task, tokenizer, args.eval_size, args.seed)
        for task in E.CORE_TASKS
    }
    items.update({
        task: E.load_transfer(task, tokenizer, args.eval_size, args.seed)
        for task in E.TRANSFER_TASKS
    })
    for task in TASKS:
        if not items[task]:
            raise RuntimeError(f"evaluation task {task} loaded no examples")
        print(f"[eval-idood] {task}: {len(items[task])}", flush=True)

    results_path = Path(args.results)
    rows = load_existing(results_path)
    done = {(row.get("arm"), row.get("task")) for row in rows}
    pending = [task for task in TASKS if (arm, task) not in done]
    if not pending:
        print(f"[eval-idood] already complete: {results_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_tag = E.load_model(
        model_name, checkpoint, torch.bfloat16, device
    )
    for task in pending:
        task_items = items[task]
        accuracy = (
            E.generation_accuracy(model, tokenizer, task_items, device)
            if task_items[0]["kind"] == "gen"
            else E.label_accuracy(model, tokenizer, task_items, device)
        )
        target_domain = TASK_DOMAIN[task]
        if source_domain == "base":
            scope = "base_reference"
        elif source_domain == target_domain:
            scope = "in_domain"
        else:
            scope = "out_of_domain"
        row = {
            "protocol": "qwen3_independent_domain_in_out_2000_v1",
            "arm": arm,
            "model": model_name,
            "method": method,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "scope": scope,
            "visit": visit,
            "task": task,
            "accuracy": accuracy,
            "n": len(task_items),
            "checkpoint": checkpoint_tag,
        }
        rows.append(row)
        atomic_rows(results_path, rows)
        print(
            f"[{arm} train={source_domain} {scope}] "
            f"{task}: {accuracy:.4f} (n={len(task_items)})",
            flush=True,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[eval-idood] complete: {results_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
