"""Shared data, answer parsing, and loss utilities for ESRL."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from fractions import Fraction

import torch
import torch.nn.functional as F


_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASH = re.compile(r"####\s*(-?[\d,\.]+)")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_ANSWER_LETTER = re.compile(
    r"(?:answer|option|choice)\D{0,8}([A-M])\b", re.I
)
_LETTER = re.compile(r"\b([A-M])\b")
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TASK = "math"


def _norm(value):
    if value is None:
        return None
    value = str(value).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = float(value)
        if not math.isfinite(number):
            return value
        return str(int(number)) if number == int(number) else str(number)
    except (ValueError, OverflowError):
        return value


def set_task(name):
    global TASK
    if name in ("mmlu", "medqa"):
        TASK = "multiple_choice"
    elif name == "math500":
        TASK = "math"
    else:
        raise ValueError(f"unsupported task: {name!r}")


def _last_boxed(text):
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    begin = start + len(marker)
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index]
    return None


def _norm_math(value):
    if value is None:
        return None
    value = str(value).strip().strip("$ ")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\,", "").replace("\\!", "").replace(" ", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "").rstrip(".。")
    fraction = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}", value)
    if fraction:
        sign = -1 if value.startswith("-\\frac") else 1
        numerator = int(fraction.group(1))
        denominator = int(fraction.group(2))
        if denominator:
            return str(Fraction(sign * numerator, denominator))
    try:
        return str(Fraction(value.replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return value


def _extract_math(text):
    boxed = _last_boxed(text)
    if boxed is not None:
        return _norm_math(boxed)
    hashes = _HASH.findall(text)
    if hashes:
        return _norm_math(hashes[-1])
    final = re.findall(
        r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*([^\n]+)", text, re.I
    )
    if final:
        return _norm_math(final[-1])
    numbers = _NUM.findall(text)
    return _norm(numbers[-1]) if numbers else None


def _extract_letter(text):
    boxed = _BOXED.findall(text)
    for candidate in reversed(boxed):
        candidate = candidate.strip().upper()
        if candidate and candidate[0] in "ABCDEFGHIJKLM":
            return candidate[0]
    matches = _ANSWER_LETTER.findall(text)
    if matches:
        return matches[-1].upper()
    matches = _LETTER.findall(text.upper())
    return matches[-1] if matches else None


def extract_answer(text):
    return _extract_letter(text) if TASK == "multiple_choice" else _extract_math(text)


def group_advantage(answers):
    valid_answers = [answer for answer in answers if answer is not None]
    if not valid_answers:
        return None, None, 0.0, False
    counts = Counter(valid_answers).most_common()
    mode, count = counts[0]
    runner_up = counts[1][1] if len(counts) > 1 else 0
    valid_mode = count >= 2 and count > runner_up
    agreement = count / len(answers)
    rewards = torch.tensor(
        [1.0 if answer == mode else 0.0 for answer in answers]
    )
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return advantages, mode, agreement, valid_mode


def frequency_filter(answers, keep, rng):
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts or max(counts.values()) == 1:
        indices = list(range(len(answers)))
        rng.shuffle(indices)
        return sorted(indices[:keep])
    frequencies = [counts.get(answer, 0) for answer in answers]
    order = sorted(range(len(answers)), key=lambda index: (-frequencies[index], index))
    return sorted(order[:keep])


@torch.no_grad()
def es_step(params, seeds, advantages, sigma, alpha, population, device):
    for seed, advantage in zip(seeds, advantages):
        coefficient = float(alpha * advantage / population)
        if coefficient == 0.0:
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        for parameter in params:
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            parameter.add_(coefficient * noise.to(parameter.dtype))


def _selected_token_logps(logits, targets, token_chunk_size=64):
    parts = []
    for begin in range(0, logits.shape[1], token_chunk_size):
        end = min(begin + token_chunk_size, logits.shape[1])
        log_probs = F.log_softmax(logits[:, begin:end, :].float(), dim=-1)
        parts.append(
            log_probs.gather(
                -1, targets[:, begin:end].unsqueeze(-1)
            ).squeeze(-1)
        )
    return torch.cat(parts, dim=1)


def sequence_logprob(model, input_ids, prompt_len, requires_grad=False, reduce="sum"):
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        logits = model(input_ids).logits
        targets = input_ids[:, 1:]
        token_logps = _selected_token_logps(logits[:, :-1, :], targets)
        completion = token_logps[:, prompt_len - 1:]
        return completion.mean() if reduce == "mean" else completion.sum()


def per_token_logp(model, input_ids, prompt_len, requires_grad=False):
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        logits = model(input_ids).logits
        targets = input_ids[:, 1:]
        token_logps = _selected_token_logps(logits[:, :-1, :], targets)
        return token_logps[:, prompt_len - 1:].squeeze(0)


def format_prompt(question, tokenizer, instruct):
    body = question + "\n\nThink step by step and put the final answer in \\boxed{}."
    if instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return body + "\nAnswer:"


def format_prompt_mmlu(question, choices, tokenizer, instruct):
    options = "\n".join(
        f"{_LETTERS[index]}. {choice}" for index, choice in enumerate(choices)
    )
    body = (
        f"{question}\n\n{options}\n\nThink step by step and put the final "
        f"answer letter (A through {_LETTERS[len(choices) - 1]}) in \\boxed{{}}."
    )
    if instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return body + "\nAnswer:"


def format_prompt_mc_eval(question, choices, tokenizer, instruct):
    options = "\n".join(
        f"{_LETTERS[index]}. {choice}" for index, choice in enumerate(choices)
    )
    body = (
        f"{question}\n\n{options}\n\nAnswer with only the single letter A through "
        f"{_LETTERS[len(choices) - 1]}."
    )
    if instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return body + "\nAnswer:"


def build_mmlu_rollout_prompt(item, tokenizer, instruct, rng):
    permutation = list(range(len(item["choices"])))
    rng.shuffle(permutation)
    shown = [item["choices"][index] for index in permutation]
    return format_prompt_mmlu(item["q"], shown, tokenizer, instruct), permutation


def canonical_letter(letter, permutation):
    if letter is None:
        return None
    position = _LETTERS.find(letter.upper())
    if position < 0 or position >= len(permutation):
        return None
    return _LETTERS[permutation[position]]


def load_math500(tokenizer, instruct, eval_size, train_size, seed=0):
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if eval_size + train_size > len(indices):
        raise ValueError("MATH-500 split request exceeds the dataset size")
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:eval_size + train_size]
    train_prompts = [
        format_prompt(dataset[index]["problem"], tokenizer, instruct)
        for index in train_indices
    ]
    eval_items = [
        {
            "prompt": format_prompt(dataset[index]["problem"], tokenizer, instruct),
            "gold": _norm_math(dataset[index]["answer"]),
        }
        for index in eval_indices
    ]
    return train_prompts, eval_items


def load_mmlu(tokenizer, instruct, eval_size, train_size, seed=0, permute=False):
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:eval_size + train_size]
    if permute:
        train_prompts = [
            {"q": dataset[index]["question"], "choices": list(dataset[index]["choices"])}
            for index in train_indices
        ]
    else:
        train_prompts = [
            format_prompt_mmlu(
                dataset[index]["question"], dataset[index]["choices"], tokenizer, instruct
            )
            for index in train_indices
        ]
    eval_items = [
        {
            "prompt": format_prompt_mc_eval(
                dataset[index]["question"], dataset[index]["choices"], tokenizer, instruct
            ),
            "gold": _LETTERS[int(dataset[index]["answer"])],
        }
        for index in eval_indices
    ]
    return train_prompts, eval_items


def _medqa_item(row):
    options = row["options"]
    if isinstance(options, dict):
        labels = [label for label in _LETTERS if str(options.get(label, "")).strip()]
        choices = [str(options[label]).strip() for label in labels]
    else:
        choices = [str(value).strip() for value in options if str(value).strip()]
        labels = list(_LETTERS[:len(choices)])
    answer = str(row["answer"]).strip().upper()
    if answer not in labels:
        return None
    return {"q": row["question"], "choices": choices, "ans": labels.index(answer)}


def load_medqa(tokenizer, instruct, eval_size, train_size, seed=0, permute=False):
    from datasets import load_dataset

    training = [
        item for row in load_dataset("mkieffer/MedQA-USMLE", split="train")
        if (item := _medqa_item(row)) is not None
    ]
    testing = [
        item for row in load_dataset("mkieffer/MedQA-USMLE", split="test")
        if (item := _medqa_item(row)) is not None
    ]
    random.Random(seed).shuffle(training)
    random.Random(seed).shuffle(testing)
    training = training[:train_size]
    testing = testing[:eval_size]
    if len(training) != train_size:
        raise ValueError("MedQA did not provide enough usable training rows")
    if permute:
        train_prompts = [
            {"q": item["q"], "choices": list(item["choices"])} for item in training
        ]
    else:
        train_prompts = [
            format_prompt_mmlu(item["q"], item["choices"], tokenizer, instruct)
            for item in training
        ]
    eval_items = [
        {
            "prompt": format_prompt_mc_eval(
                item["q"], item["choices"], tokenizer, instruct
            ),
            "gold": _LETTERS[item["ans"]],
        }
        for item in testing
    ]
    return train_prompts, eval_items


def load_task(dataset, tokenizer, instruct, eval_size, train_size, seed=0, permute=False):
    set_task(dataset)
    if dataset == "math500":
        return load_math500(tokenizer, instruct, eval_size, train_size, seed)
    if dataset == "mmlu":
        return load_mmlu(
            tokenizer, instruct, eval_size, train_size, seed, permute
        )
    if dataset == "medqa":
        return load_medqa(
            tokenizer, instruct, eval_size, train_size, seed, permute
        )
    raise ValueError(f"unsupported dataset: {dataset!r}")
