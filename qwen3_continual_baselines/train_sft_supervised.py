"""Ground-truth supervised fine-tuning baseline for the ES-DGRPO harness.

This is conventional SFT, not ``train.py --method sft_good``:

* MATH-500 trains on the reference solution and gold final answer.
* MMLU and MedQA-USMLE train on the gold answer letter.
* Only completion tokens contribute to the cross-entropy loss.

The data split, prompt format, LoRA configuration, snapshot names, optimizer
state, and resume flags match the existing trainers, allowing unchanged use of
``robust_train.py`` and ``eval_continual.py``.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch

import common as C


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_student(model_name, dtype, device, lora_r):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, config).to(device)


def load_adapter_into(model, path):
    from safetensors.torch import load_file

    adapter_file = os.path.join(path, "adapter_model.safetensors")
    if not os.path.isfile(adapter_file):
        raise RuntimeError(f"no adapter_model.safetensors in {path}")

    saved = load_file(adapter_file)
    current = model.state_dict()
    loaded = 0
    with torch.no_grad():
        for key, value in saved.items():
            model_key = key.replace(
                "lora_A.weight", "lora_A.default.weight"
            ).replace(
                "lora_B.weight", "lora_B.default.weight"
            )
            if model_key in current:
                current[model_key].copy_(
                    value.to(current[model_key].device, current[model_key].dtype)
                )
                loaded += 1
    if loaded == 0:
        raise RuntimeError(f"no LoRA tensors from {path} matched the model")
    print(f"[sft] resumed LoRA from {path} ({loaded} tensors)", flush=True)


def restore_optimizer(optimizer, path, device):
    optimizer_file = os.path.join(path, "optimizer.pt")
    if not os.path.isfile(optimizer_file):
        print(
            f"[sft] WARNING: no optimizer.pt in {path}; using a fresh optimizer",
            flush=True,
        )
        return
    try:
        optimizer.load_state_dict(torch.load(optimizer_file, map_location=device))
        print(f"[sft] restored optimizer from {optimizer_file}", flush=True)
    except Exception as exc:
        print(
            f"[sft] WARNING: optimizer state could not be restored ({exc}); "
            "using a fresh optimizer",
            flush=True,
        )


def save_snapshot(model, optimizer, out, visits):
    path = os.path.join(out, f"snap{visits:06d}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    return path


def split_indices(size, eval_size, train_size, seed):
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[eval_size:eval_size + train_size]


def gsm8k_target(answer):
    """Keep the reference reasoning but align its final format with the prompt."""
    gold = C.gsm8k_gold(answer)
    if gold is None:
        raise ValueError(f"could not parse GSM8K gold answer: {answer!r}")
    reasoning = answer.rsplit("####", 1)[0].strip()
    if reasoning:
        return f"{reasoning}\nThe final answer is \\boxed{{{gold}}}."
    return f"The final answer is \\boxed{{{gold}}}."


def load_labeled_examples(dataset, tok, instruct, eval_size, train_size, seed):
    """Use exactly the same shuffled held-out/train partition as common.py."""
    from datasets import load_dataset

    if dataset == "math500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        indices = split_indices(len(data), eval_size, train_size, seed)
        return [
            {
                "kind": "math",
                "prompt": C.format_prompt(data[i]["problem"], tok, instruct),
                "target": (
                    f"{data[i]['solution'].strip()}\n"
                    f"The final answer is \\boxed{{{data[i]['answer']}}}."
                ),
            }
            for i in indices
        ]

    if dataset == "mmlu":
        data = load_dataset("cais/mmlu", "all", split="test")
        indices = split_indices(len(data), eval_size, train_size, seed)
        return [
            {
                "kind": "mc",
                "question": data[i]["question"],
                "choices": list(data[i]["choices"]),
                "answer": int(data[i]["answer"]),
            }
            for i in indices
        ]

    if dataset == "medqa":
        data = load_dataset("mkieffer/MedQA-USMLE", split="train")
        clean = []
        for row in data:
            item = C._medqa_item(row)
            if item is not None:
                clean.append(
                    {
                        "question": item["q"],
                        "choices": item["choices"],
                        "answer": item["ans"],
                    }
                )
        indices = list(range(len(clean)))
        random.Random(seed).shuffle(indices)
        indices = indices[:train_size]
        return [dict(kind="mc", **clean[i]) for i in indices]

    raise ValueError(f"SFT is not implemented for dataset {dataset!r}")


def materialize(example, tok, instruct, permute, seed, global_visit):
    if example["kind"] == "math":
        return example["prompt"], example["target"]

    choices = list(example["choices"])
    answer = int(example["answer"])
    if permute:
        order = list(range(len(choices)))
        # Key the augmentation to the visit so crash-resume is exactly reproducible.
        random.Random(seed * 1_000_003 + global_visit * 7_919 + 17).shuffle(order)
        displayed = [choices[index] for index in order]
        displayed_answer = order.index(answer)
    else:
        displayed = choices
        displayed_answer = answer

    prompt = C.format_prompt_mmlu(example["question"], displayed, tok, instruct)
    target = f"The final answer is \\boxed{{{LETTERS[displayed_answer]}}}."
    return prompt, target


def encode_example(tok, prompt, target, max_length, device):
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    target_ids = tok.encode(target + tok.eos_token, add_special_tokens=False)
    total = len(prompt_ids) + len(target_ids)
    if total > max_length:
        raise RuntimeError(
            f"SFT example has {total} tokens, exceeding --max-length={max_length}; "
            "increase the limit rather than silently truncating the gold target"
        )

    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    labels = torch.tensor(
        [[-100] * len(prompt_ids) + target_ids], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels, len(target_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--dataset", default="math500",
                        choices=["math500", "mmlu", "medqa"])
    parser.add_argument("--permute-choices", action="store_true")
    parser.add_argument("--out", default="out_sft/sft_supervised")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--snap-every", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--start-visit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    instruct = "Instruct" in args.model

    examples = load_labeled_examples(
        args.dataset, tok, instruct, args.eval_size, args.train_size, args.seed
    )
    if len(examples) != args.train_size:
        raise SystemExit(
            f"requested {args.train_size} examples but {args.dataset} supplied "
            f"{len(examples)} after the held-out split"
        )

    model = build_student(args.model, dtype, device, args.lora_r)
    if args.init_adapter:
        load_adapter_into(model, args.init_adapter)
    model.train()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    if args.init_adapter:
        restore_optimizer(optimizer, args.init_adapter, device)

    expected_visits = len(examples) * args.epochs
    if args.start_visit < 0 or args.start_visit > expected_visits:
        raise SystemExit(
            f"--start-visit must be between 0 and {expected_visits}, "
            f"got {args.start_visit}"
        )

    print(
        f"[sft] SUPERVISED gold-label training | dataset={args.dataset} | "
        f"examples={len(examples)} x epochs={args.epochs} | lr={args.lr} | "
        f"LoRA r={args.lora_r}",
        flush=True,
    )
    print(
        "[sft] no rollouts, majority vote, reward model, or RL objective",
        flush=True,
    )

    visits = args.start_visit
    start_time = time.time()
    for epoch in range(args.epochs):
        for local_index, example in enumerate(examples):
            global_index = epoch * len(examples) + local_index
            if global_index < args.start_visit:
                continue

            prompt, target = materialize(
                example,
                tok,
                instruct,
                args.permute_choices and args.dataset in ("mmlu", "medqa"),
                args.seed,
                global_index,
            )
            input_ids, attention_mask, labels, target_tokens = encode_example(
                tok, prompt, target, args.max_length, device
            )

            optimizer.zero_grad()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            visits += 1
            if visits % 25 == 0:
                print(
                    f"  [{visits}] loss={float(loss.detach()):.4f} "
                    f"target_tokens={target_tokens}",
                    flush=True,
                )
            if args.snap_every and visits % args.snap_every == 0:
                path = save_snapshot(model, optimizer, args.out, visits)
                print(f"  [snap] {visits} visits -> {path}", flush=True)

            del output, loss, input_ids, attention_mask, labels

    model.save_pretrained(args.out)
    if args.snap_every and visits % args.snap_every:
        save_snapshot(model, optimizer, args.out, visits)
    print(
        f"[sft] done: {visits}/{expected_visits} visits in "
        f"{(time.time() - start_time) / 60:.1f} minutes; saved -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
