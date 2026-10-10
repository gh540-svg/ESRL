"""Evaluate Qwen3 continual checkpoints on retention and transfer tasks.

Core retention tasks are scored at every saved checkpoint.  The six larger
transfer tasks are scored for the base model and at each 2,000-visit stage
boundary.  Results are appended atomically so evaluation can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import string

import torch
import torch.nn.functional as F

import common as C


LETTERS = string.ascii_uppercase
CORE_TASKS = ("math500_holdout", "mmlu_holdout", "medqa_test")
TRANSFER_TASKS = (
    "amc23",
    "aime25",
    "arc_challenge",
    "gpqa_diamond",
    "medmcqa_validation",
    "pubmedqa_labeled",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def prompt_math(question, tok):
    return C.format_prompt(str(question), tok, False)


def prompt_mc(question, choices, tok):
    return C.format_prompt_mc_eval(
        str(question), [str(choice) for choice in choices], tok, False
    )


def mc_item(question, choices, answer_index, tok):
    choices = list(choices)
    if not 0 <= int(answer_index) < len(choices):
        return None
    return {
        "prompt": prompt_mc(question, choices, tok),
        "gold": LETTERS[int(answer_index)],
        "kind": "mc",
        "labels": list(LETTERS[:len(choices)]),
    }


def shuffled(rows, size, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:size] if size else rows


def load_core(task, tok, size, seed):
    if task == "math500_holdout":
        _, rows = C.load_math500(tok, False, 100, 400, seed)
        return [dict(row, kind="gen", answer_task="math500") for row in rows]
    if task == "mmlu_holdout":
        _, rows = C.load_mmlu(tok, False, size, 500, seed, False)
        return [
            dict(row, kind="mc", labels=list("ABCD"), answer_task="mmlu")
            for row in rows
        ]
    if task == "medqa_test":
        _, rows = C.load_medqa(tok, False, size, 500, seed, False)
        return [
            dict(
                row,
                kind="mc",
                answer_task="medqa",
            )
            for row in rows
        ]
    raise ValueError(task)


def load_amc23(tok):
    from datasets import load_dataset

    dataset = load_dataset("rawsh/amc23", split="train")
    items = []
    for row in dataset:
        prompt = row.get("prompt")
        if isinstance(prompt, list) and prompt:
            question = prompt[0].get("content", "")
        else:
            question = row.get("question") or row.get("problem") or str(prompt)
        reward = row.get("reward_model", {})
        gold = reward.get("ground_truth") if isinstance(reward, dict) else None
        if gold is None:
            gold = row.get("answer")
        items.append(
            {
                "prompt": prompt_math(question, tok),
                "gold": C._norm_math(gold),
                "kind": "gen",
                "answer_task": "amc23",
            }
        )
    return items


def load_aime25(tok):
    from datasets import load_dataset

    dataset = load_dataset("yentinglin/aime_2025", split="train")
    return [
        {
            "prompt": prompt_math(row["problem"], tok),
            "gold": C._norm_math(row["answer"]),
            "kind": "gen",
            "answer_task": "aime25",
        }
        for row in dataset
    ]


def load_arc_challenge(tok, size, seed):
    from datasets import load_dataset

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in shuffled(dataset, 0, seed):
        labels = list(row["choices"]["label"])
        options = list(row["choices"]["text"])
        if row["answerKey"] not in labels:
            continue
        item = mc_item(row["question"], options, labels.index(row["answerKey"]), tok)
        if item:
            item["answer_task"] = "arc_challenge"
            items.append(item)
        if len(items) == size:
            break
    return items


def _pick(row, *keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    raise KeyError(f"none of {keys!r} found in GPQA row")


def load_gpqa(tok, seed):
    from datasets import load_dataset

    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for index, row in enumerate(dataset):
        question = _pick(row, "Question", "question")
        correct = _pick(row, "Correct Answer", "correct_answer")
        wrong = [
            _pick(row, f"Incorrect Answer {number}", f"incorrect_answer{number}")
            for number in range(1, 4)
        ]
        choices = [correct] + wrong
        order = list(range(4))
        random.Random(seed * 1_000_003 + index).shuffle(order)
        item = mc_item(question, [choices[position] for position in order], order.index(0), tok)
        if item:
            item["answer_task"] = "gpqa"
            items.append(item)
    return items


def load_medmcqa(tok, size, seed):
    from datasets import load_dataset

    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    items = []
    for row in shuffled(dataset, size, seed):
        item = mc_item(
            row["question"],
            [row["opa"], row["opb"], row["opc"], row["opd"]],
            int(row["cop"]),
            tok,
        )
        if item:
            item["answer_task"] = "medmcqa"
            items.append(item)
    return items


def load_pubmedqa(tok, size, seed):
    from datasets import load_dataset

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = shuffled(dataset, size, seed)
    items = []
    for row in rows:
        context = row.get("context", {})
        contexts = context.get("contexts", []) if isinstance(context, dict) else []
        body = "Context:\n" + "\n".join(str(value) for value in contexts)
        body += f"\n\nQuestion: {row['question']}\nAnswer yes, no, or maybe."
        items.append(
            {
                "prompt": body + "\nAnswer:",
                "gold": str(row["final_decision"]).lower(),
                "kind": "label",
                "labels": ["yes", "no", "maybe"],
                "answer_task": "pubmedqa",
            }
        )
    return items


def load_transfer(task, tok, size, seed):
    if task == "amc23":
        return load_amc23(tok)
    if task == "aime25":
        return load_aime25(tok)
    if task == "arc_challenge":
        return load_arc_challenge(tok, size, seed)
    if task == "gpqa_diamond":
        return load_gpqa(tok, seed)
    if task == "medmcqa_validation":
        return load_medmcqa(tok, size, seed)
    if task == "pubmedqa_labeled":
        return load_pubmedqa(tok, size, seed)
    raise ValueError(task)


def load_model(model_name, checkpoint, dtype, device):
    from transformers import AutoModelForCausalLM

    checkpoint = Path(checkpoint) if checkpoint else None
    if checkpoint and (checkpoint / "config.json").is_file() and not (
        checkpoint / "adapter_config.json"
    ).is_file():
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device)
        tag = str(checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        tag = "base"
        if checkpoint:
            from safetensors.torch import load_file

            config = json.loads((checkpoint / "adapter_config.json").read_text())
            scale = config["lora_alpha"] / config["r"]
            base_state = model.state_dict()
            adapter = load_file(checkpoint / "adapter_model.safetensors")
            loaded = 0
            for key, a_value in adapter.items():
                if not key.endswith("lora_A.weight"):
                    continue
                prefix = key[: -len("lora_A.weight")]
                b_value = adapter.get(prefix + "lora_B.weight")
                weight_key = prefix.replace("base_model.model.", "") + "weight"
                if b_value is None or weight_key not in base_state:
                    continue
                with torch.no_grad():
                    delta = b_value.to(device, torch.float32) @ a_value.to(device, torch.float32)
                    base_state[weight_key].add_((delta * scale).to(base_state[weight_key].dtype))
                loaded += 1
            if loaded == 0:
                raise RuntimeError(f"no LoRA tensors from {checkpoint} matched {model_name}")
            tag = str(checkpoint)
    model.eval()
    return model, tag


@torch.no_grad()
def label_accuracy(model, tok, items, device, batch_size=8):
    old_padding, old_truncation = tok.padding_side, tok.truncation_side
    tok.padding_side, tok.truncation_side = "left", "left"
    correct = 0
    candidate_cache = {}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        encoded = tok(
            [item["prompt"] for item in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)
        logp = F.log_softmax(model(**encoded).logits[:, -1, :].float(), dim=-1)
        for row_index, item in enumerate(chunk):
            best, best_score = None, -float("inf")
            for label in item["labels"]:
                key = str(label)
                if key not in candidate_cache:
                    ids = set(tok.encode(key, add_special_tokens=False)[:1])
                    ids.update(tok.encode(" " + key, add_special_tokens=False)[:1])
                    candidate_cache[key] = sorted(ids)
                score = max(float(logp[row_index, token]) for token in candidate_cache[key])
                if score > best_score:
                    best, best_score = key, score
            if str(best).lower() == str(item["gold"]).lower():
                correct += 1
    tok.padding_side, tok.truncation_side = old_padding, old_truncation
    return correct / len(items)


@torch.no_grad()
def generation_accuracy(model, tok, items, device):
    correct = 0
    for start in range(0, len(items), 4):
        chunk = items[start:start + 4]
        rollouts, prompt_lengths = C.batched_generate(
            model,
            tok,
            [item["prompt"] for item in chunk],
            0.0,
            512,
            device,
            greedy=True,
        )
        for rollout, prompt_length, item in zip(rollouts, prompt_lengths, chunk):
            C.set_task(item["answer_task"])
            completion = tok.decode(
                rollout[0, prompt_length:], skip_special_tokens=True
            )
            prediction = C.extract_answer(completion)
            if C.answers_match(prediction, item["gold"]):
                correct += 1
    return correct / len(items)


def checkpoints(manifest, arm):
    visits_per_stage = int(manifest["visits_per_stage"])
    output = []
    for stage in manifest["methods"][arm]:
        directory = Path(stage["dir"])
        if not directory.is_dir():
            continue
        for snapshot in sorted(directory.glob("snap[0-9]*")):
            adapter_ok = (
                (snapshot / "adapter_config.json").is_file()
                and (snapshot / "adapter_model.safetensors").is_file()
            )
            full_ok = (snapshot / "config.json").is_file() and (
                (snapshot / "model.safetensors").is_file()
                or (snapshot / "pytorch_model.bin").is_file()
                or (snapshot / "model.safetensors.index.json").is_file()
                or (snapshot / "pytorch_model.bin.index.json").is_file()
            )
            if not (adapter_ok or full_ok):
                continue
            local_visit = int("".join(c for c in snapshot.name if c.isdigit()))
            global_visit = (int(stage["stage"]) - 1) * visits_per_stage + local_visit
            output.append(
                (global_visit, int(stage["stage"]), stage["dataset"], snapshot)
            )
    return sorted(output)


def load_rows(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    except (OSError, ValueError):
        return []


def save_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    arm = next(iter(manifest["methods"]))
    model_name = manifest["model"]
    visits_per_stage = int(manifest["visits_per_stage"])
    results_path = Path(args.results)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    items = {
        task: load_core(task, tok, args.eval_size, args.seed) for task in CORE_TASKS
    }
    items.update(
        {
            task: load_transfer(task, tok, args.eval_size, args.seed)
            for task in TRANSFER_TASKS
        }
    )
    for task, task_items in items.items():
        if not task_items:
            raise RuntimeError(f"evaluation task {task} loaded no items")
        print(f"[eval] {task}: {len(task_items)}", flush=True)

    rows = load_rows(results_path)
    done = {(row["arm"], row["visit"], row["task"]) for row in rows}
    work = [(0, 0, "base", None)] + checkpoints(manifest, arm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for visit, stage, trained_on, checkpoint in work:
        tasks = list(CORE_TASKS)
        if visit == 0 or visit % visits_per_stage == 0:
            tasks.extend(TRANSFER_TASKS)
        tasks = [task for task in tasks if (arm, visit, task) not in done]
        if not tasks:
            continue
        model, tag = load_model(model_name, checkpoint, torch.bfloat16, device)
        for task in tasks:
            task_items = items[task]
            kind = task_items[0]["kind"]
            accuracy = (
                generation_accuracy(model, tok, task_items, device)
                if kind == "gen"
                else label_accuracy(model, tok, task_items, device)
            )
            row = {
                "arm": arm,
                "model": model_name,
                "method": manifest["method"],
                "visit": visit,
                "stage": stage,
                "trained_on": trained_on,
                "task": task,
                "accuracy": accuracy,
                "n": len(task_items),
                "checkpoint": tag,
            }
            rows.append(row)
            save_rows(results_path, rows)
            print(
                f"[{arm} v{visit} s{stage}/{trained_on}] {task}: {accuracy:.4f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[eval] complete: {results_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
