"""Fast offline validation before any GPU job is submitted."""
from __future__ import annotations

import argparse

import common as C
import eval_continual_qwen3 as evaluation


MODELS = {
    "0p6b": "Qwen/Qwen3-0.6B-Base",
    "4b": "Qwen/Qwen3-4B-Base",
    "8b": "Qwen/Qwen3-8B-Base",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=tuple(MODELS), default="4b", nargs="?")
    args = parser.parse_args()

    import transformers
    from transformers import AutoConfig, AutoTokenizer

    major_minor = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    if major_minor < (4, 51):
        raise SystemExit(f"transformers {transformers.__version__} is too old for Qwen3")

    model = MODELS[args.model]
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[ok] {model}: model_type={config.model_type}")

    expected = {
        "math500": (400, 100),
        "mmlu": (500, 200),
        "medqa": (500, 200),
    }
    for dataset, (train_count, eval_count) in expected.items():
        train, heldout = C.load_task(
            dataset,
            tokenizer,
            False,
            eval_count,
            train_count,
            seed=0,
            permute=dataset in ("mmlu", "medqa"),
        )
        assert len(train) == train_count, (dataset, len(train), train_count)
        assert len(heldout) == eval_count, (dataset, len(heldout), eval_count)
        print(f"[ok] {dataset}: train={len(train)} eval={len(heldout)}")

    tasks = list(evaluation.CORE_TASKS) + list(evaluation.TRANSFER_TASKS)
    for task in tasks:
        rows = (
            evaluation.load_core(task, tokenizer, 200, 0)
            if task in evaluation.CORE_TASKS
            else evaluation.load_transfer(task, tokenizer, 200, 0)
        )
        if not rows:
            raise RuntimeError(f"{task} loaded zero rows")
        print(f"[ok] {task}: {len(rows)}")
    print("[ok] all cached assets and dataset schemas passed")


if __name__ == "__main__":
    main()
