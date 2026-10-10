"""Focused PEFT model construction and updates for ESRL and Mode-SFT."""

from __future__ import annotations

import os

import torch

import common as C
import model_compat as MC


def build_models(args, dtype, device):
    """Build one LoRA student and, when requested, a frozen base reference."""
    from peft import LoraConfig, get_peft_model

    tokenizer = MC.load_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = MC.load_text_generation_model(args.model, torch_dtype=dtype).to(device)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules=MC.lora_target_modules(args.model),
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(base, config).to(device)

    if args.init_adapter:
        from safetensors.torch import load_file

        weights = load_file(
            os.path.join(args.init_adapter, "adapter_model.safetensors")
        )
        state = student.state_dict()
        loaded = 0
        with torch.no_grad():
            for name, value in weights.items():
                target = name.replace(
                    "lora_A.weight", "lora_A.default.weight"
                ).replace("lora_B.weight", "lora_B.default.weight")
                if target in state:
                    state[target].copy_(value.to(state[target].device, state[target].dtype))
                    loaded += 1
        if loaded == 0:
            raise RuntimeError(
                f"no LoRA tensors matched the adapter in {args.init_adapter}"
            )
        print(f"[policy] restored {loaded} LoRA tensors from {args.init_adapter}")

    reference = None
    if args.kl_lambda > 0:
        reference = MC.load_text_generation_model(
            args.model, torch_dtype=dtype
        ).to(device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    return tokenizer, student, None, reference


def _prompt_len(prompt_lens, index):
    if isinstance(prompt_lens, (list, tuple)):
        return int(prompt_lens[index])
    return int(prompt_lens)


def update(
    args,
    method,
    student,
    optimizer,
    rollouts,
    answers,
    advantages,
    mode,
    reference_logps,
    seeds,
    prompt_lens,
    device,
    reference=None,
    old_lp=None,
    zero_grad=True,
    step=True,
    loss_scale=1.0,
):
    """Apply ESRL's group-relative policy-gradient and optional KL update."""
    del answers, mode, reference_logps, seeds
    if method != "esrl":
        raise ValueError(f"unsupported policy update: {method!r}")
    if old_lp:
        raise ValueError("ESRL's published configuration does not use PPO ratios")

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    total = 0.0
    population = len(advantages)
    for index in range(population):
        prompt_len = _prompt_len(prompt_lens, index)
        reference_token_logps = None
        if args.kl_lambda > 0:
            if reference is None:
                raise RuntimeError("KL regularization requires a frozen reference model")
            reference_token_logps = C.per_token_logp(
                reference, rollouts[index], prompt_len, False
            )

        if args.kl_lambda > 0:
            student_token_logps = C.per_token_logp(
                student, rollouts[index], prompt_len, True
            )
            policy_logp = student_token_logps.mean()
        else:
            student_token_logps = None
            policy_logp = C.sequence_logprob(
                student, rollouts[index], prompt_len, True, "mean"
            )

        loss = -(advantages[index].to(device) * policy_logp) / population
        if args.kl_lambda > 0:
            delta = reference_token_logps - student_token_logps
            kl = (torch.exp(delta) - delta - 1.0).mean()
            loss = loss + args.kl_lambda * kl / population
        loss = loss * float(loss_scale)
        loss.backward()
        total += float(loss.detach())

    if step:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    return total


def mode_sft_update(student, optimizer, rollouts, prompt_lens, answers, mode):
    """Apply SFT only to rollouts matching the unique modal pseudo-label."""
    selected = [index for index, answer in enumerate(answers) if answer == mode]
    if not selected:
        return None
    optimizer.zero_grad(set_to_none=True)
    scale = 1.0 / len(selected)
    total = 0.0
    for index in selected:
        loss = -C.sequence_logprob(
            student,
            rollouts[index],
            _prompt_len(prompt_lens, index),
            True,
            "mean",
        )
        (loss * scale).backward()
        total += float(loss.detach())
    trainable = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    return total * scale
