"""ESRL: LoRA-perturbed teachers with a group-relative student update.

ESRL perturbs the current rank-16 LoRA policy inside persistent vLLM, produces
one rollout from each of 16 perturbed teachers, and updates a differentiable
student with group-relative advantages and optional base-policy KL anchoring.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch

import common as C
import esrl_core as core
from remote_vllm_engine import adapter_sync_parent
from vllm_utils import request_to_sequence
from vllm_adapter import (
    adapter_scale,
    create_initial_peft_adapter,
    rpc_one,
    validate_adapter,
    vllm_version_check,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument(
        "--dataset", required=True, choices=("math500", "mmlu", "medqa")
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--eval-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--rollouts-per-member", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--max-new", type=int, default=8196)
    parser.add_argument("--p-high", type=float, default=0.95)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-lambda", type=float, default=0.001)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--freq-filter", type=float, default=0.0)
    parser.add_argument("--snap-every", type=int, default=500)
    parser.add_argument(
        "--prompt-update-batch-size", type=int, default=1,
        help=("number of prompt groups whose losses are averaged into one "
              "optimizer step; 1 preserves per-prompt updates"),
    )
    parser.add_argument(
        "--snap-every-updates", type=int, default=0,
        help=("save after this many optimizer steps (0 disables update-based "
              "snapshots); directory names retain cumulative prompt visits"),
    )
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--start-visit", type=int, default=0)
    parser.add_argument(
        "--max-visits", type=int, default=None,
        help="stop after exactly this many total prompt visits",
    )
    parser.add_argument("--permute-choices", action="store_true")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("float16", "bfloat16")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--rollout-prompt-batch-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=12292)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.population < 2:
        parser.error("--population must be at least 2")
    if args.rollouts_per_member != 1:
        parser.error("ESRL uses one rollout per perturbed adapter")
    for name in (
        "block_size", "prompt_update_batch_size", "rollout_prompt_batch_size", "max_model_len",
        "max_num_seqs", "lora_r",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    simultaneous = args.population * args.rollout_prompt_batch_size
    if args.max_num_seqs < simultaneous:
        parser.error(
            f"--max-num-seqs must be >= population*prompt-batch ({simultaneous})"
        )
    if args.sigma < 0 or args.kl_lambda < 0:
        parser.error("--sigma and --kl-lambda must be non-negative")
    if not 0.0 <= args.freq_filter <= 1.0:
        parser.error("--freq-filter must be in [0, 1]")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.seed < 0 or (args.data_seed is not None and args.data_seed < 0):
        parser.error("--seed and --data-seed must be non-negative")
    if args.snap_every_updates < 0:
        parser.error("--snap-every-updates must be non-negative")
    if args.block_size % args.prompt_update_batch_size:
        parser.error("--block-size must be divisible by --prompt-update-batch-size")
    return args


def _build_member_prompts(
    block_prompts, seeds, permute, tokenizer, instruct,
):
    """Create one deterministic MMLU permutation stream per ensemble member."""
    if not permute:
        return (
            [[item for item in block_prompts] for _ in seeds],
            [[None for _ in block_prompts] for _ in seeds],
        )
    member_prompts, member_permutations = [], []
    for seed in seeds:
        rng = random.Random(int(seed) * 7919 + 1)
        built = [
            C.build_mmlu_rollout_prompt(item, tokenizer, instruct, rng)
            for item in block_prompts
        ]
        member_prompts.append([value[0] for value in built])
        member_permutations.append([value[1] for value in built])
    return member_prompts, member_permutations


def main():
    args = parse_args()
    data_seed = args.seed if args.data_seed is None else args.data_seed
    os.environ["VLLM_USE_V1"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    version = vllm_version_check()

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    if os.environ.get("VLLM_REMOTE_HOST"):
        from remote_vllm_engine import RemoteVLLMEngine as LLM
    else:
        from vllm import LLM

    torch.manual_seed(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.init_adapter).resolve() if args.init_adapter else None
    if source is not None:
        adapter_dir = source
    else:
        adapter_dir = (output / ".esrl_initial_adapter").resolve()
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            create_initial_peft_adapter(adapter_dir, args.model, args.lora_r, args.seed)
    validate_adapter(adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    simultaneous = args.population * args.rollout_prompt_batch_size
    engine = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        enable_lora=True,
        max_loras=args.population,
        max_cpu_loras=args.population,
        max_lora_rank=args.lora_r,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
        worker_extension_cls="vllm_es_worker.VLLMESWorkerExtension",
    )
    requests = [
        LoRARequest(f"esrl_member_{member}", member + 1, str(adapter_dir))
        for member in range(args.population)
    ]
    engine.generate(
        ["Warm up."] * args.population,
        [
            SamplingParams(temperature=0.0, max_tokens=1, seed=args.seed + member)
            for member in range(args.population)
        ],
        lora_request=requests,
        use_tqdm=False,
    )
    rpc_one(
        engine, "es_initialize",
        [request.lora_int_id for request in requests],
        str(adapter_dir), adapter_scale(adapter_dir),
    )

    # The vLLM adapters supply the perturbed teachers. The training node owns
    # only the differentiable student and optional frozen KL reference.
    args.method = "esrl"
    args.clip_eps = 0.2
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    _, student, _, reference = core.build_models(args, dtype, device)
    student.train()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    if source is not None:
        optimizer_path = source / "optimizer.pt"
        if optimizer_path.is_file():
            optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=device))
            print(f"[esrl] restored optimizer from {optimizer_path}")
        else:
            print(
                f"[esrl] WARNING: {optimizer_path} absent; fresh AdamW",
                flush=True,
            )

    instruct = "Instruct" in args.model
    permute = args.permute_choices and args.dataset in ("mmlu", "medqa")
    train_prompts, _ = C.load_task(
        args.dataset, tokenizer, instruct, args.eval_size,
        args.train_size, data_seed, permute,
    )
    expected_visits = len(train_prompts) * args.epochs
    if not 0 <= args.start_visit <= expected_visits:
        raise SystemExit(
            f"--start-visit must be in [0, {expected_visits}], got {args.start_visit}"
        )
    if args.start_visit % args.block_size:
        raise SystemExit("--start-visit must align with --block-size")
    if args.max_visits is not None and not (
        args.start_visit < args.max_visits <= expected_visits
    ):
        raise SystemExit(
            "--max-visits must be greater than --start-visit and no larger "
            "than train-size * epochs"
        )
    target_visits = (
        args.max_visits if args.max_visits is not None else expected_visits
    )

    print(
        f"[esrl] vLLM={version}; model={args.model}; "
        f"population={args.population}; simultaneous_rollouts<={simultaneous}; "
        f"behavior_perturbation=LoRA-only sigma={args.sigma:g}", flush=True,
    )
    print(
        "[esrl] LoRA-perturbed teacher ensemble with group-relative update.",
        flush=True,
    )

    scratch_parent = adapter_sync_parent()
    sync_root = Path(tempfile.mkdtemp(
        prefix=f"esrl_{os.environ.get('SLURM_JOB_ID', 'local')}_",
        dir=scratch_parent,
    ))
    atexit.register(shutil.rmtree, sync_root, ignore_errors=True)
    sync_dir = sync_root / "policy"
    sync_scope = "shared" if os.environ.get("VLLM_REMOTE_HOST") else "node-local"
    print(f"[esrl] {sync_scope} adapter sync: {sync_dir}", flush=True)

    def save_checkpoint(path):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(destination)
        torch.save(optimizer.state_dict(), destination / "optimizer.pt")

    def synchronize_and_perturb(seeds):
        if sync_dir.exists():
            for candidate in sync_dir.glob("adapter_model.*"):
                candidate.unlink()
        student.save_pretrained(sync_dir)
        validate_adapter(sync_dir)
        rpc_one(
            engine, "es_initialize",
            [request.lora_int_id for request in requests],
            str(sync_dir), adapter_scale(sync_dir),
        )
        rpc_one(engine, "es_materialize", seeds, args.sigma)

    visits = args.start_visit
    update_steps = args.start_visit // args.prompt_update_batch_size
    used = skipped = 0
    started = time.monotonic()
    blocks_per_epoch = (len(train_prompts) + args.block_size - 1) // args.block_size
    start_block = args.start_visit // args.block_size

    for epoch in range(args.epochs):
        for block_index in range(blocks_per_epoch):
            if visits >= target_visits:
                break
            global_block = epoch * blocks_per_epoch + block_index
            if global_block < start_block:
                continue
            block_prompts = train_prompts[
                block_index * args.block_size:(block_index + 1) * args.block_size
            ]
            block_prompts = block_prompts[:target_visits - visits]
            seeds = [10_000 * global_block + member for member in range(args.population)]
            synchronize_and_perturb(seeds)
            member_prompts, member_permutations = _build_member_prompts(
                block_prompts, seeds, permute, tokenizer, instruct
            )
            per_prompt = []
            prompt_batch = min(args.rollout_prompt_batch_size, len(block_prompts))

            for chunk_start in range(0, len(block_prompts), prompt_batch):
                chunk_end = min(chunk_start + prompt_batch, len(block_prompts))
                flat_prompts, flat_permutations, flat_requests = [], [], []
                group_ranges = []
                for prompt_index in range(chunk_start, chunk_end):
                    begin = len(flat_prompts)
                    for member in range(args.population):
                        flat_prompts.append(member_prompts[member][prompt_index])
                        flat_permutations.append(
                            member_permutations[member][prompt_index]
                        )
                        flat_requests.append(requests[member])
                    group_ranges.append((begin, len(flat_prompts)))

                sampling = []
                for local_index in range(len(flat_prompts)):
                    sampling.append(SamplingParams(
                        temperature=args.temp,
                        top_p=0.95,
                        max_tokens=args.max_new,
                        seed=(
                            args.seed * 1_000_003
                            + global_block * args.block_size * args.population
                            + chunk_start * args.population
                            + local_index
                        ),
                    ))
                generated = engine.generate(
                    flat_prompts,
                    sampling,
                    lora_request=flat_requests,
                    use_tqdm=False,
                )
                flat_rollouts, flat_plens, flat_answers = [], [], []
                for result, permutation in zip(generated, flat_permutations):
                    sequence, prompt_len = request_to_sequence(
                        result, tokenizer.eos_token_id
                    )
                    # vLLM returns token IDs through CPU-side request objects.
                    # Differentiable scoring expects the rollout tensor
                    # on the same device as the differentiable PEFT student.
                    sequence = sequence.to(device=device, non_blocking=True)
                    answer = C.extract_answer(result.outputs[0].text)
                    if permutation is not None:
                        answer = C.canonical_letter(answer, permutation)
                    flat_rollouts.append(sequence)
                    flat_plens.append(prompt_len)
                    flat_answers.append(answer)
                for begin, end in group_ranges:
                    per_prompt.append({
                        "rollouts": flat_rollouts[begin:end],
                        "plens": flat_plens[begin:end],
                        "answers": flat_answers[begin:end],
                    })

            for update_start in range(
                0, len(per_prompt), args.prompt_update_batch_size
            ):
                update_batch = per_prompt[
                    update_start:update_start + args.prompt_update_batch_size
                ]
                prepared = []
                batch_skipped = 0
                for local_offset, original in enumerate(update_batch):
                    prompt_offset = update_start + local_offset
                    data = original
                    if args.freq_filter > 0:
                        keep = max(
                            2, int(round(len(data["answers"]) * args.freq_filter))
                        )
                        indices = C.frequency_filter(
                            data["answers"], keep,
                            random.Random(
                                917_653 + 1000 * global_block + prompt_offset
                            ),
                        )
                        data = {
                            key: [values[index] for index in indices]
                            for key, values in data.items()
                        }
                    advantage, mode, agreement, valid = C.group_advantage(
                        data["answers"]
                    )
                    if not valid or agreement > args.p_high:
                        skipped += 1
                        batch_skipped += 1
                        continue
                    prepared.append((data, advantage, mode))
                    used += 1

                visits += len(update_batch)
                loss = None
                if prepared:
                    optimizer.zero_grad(set_to_none=True)
                    loss = 0.0
                    prompt_scale = 1.0 / len(prepared)
                    for data, advantage, mode in prepared:
                        loss += core.update(
                            args, "esrl", student, optimizer,
                            data["rollouts"], data["answers"], advantage, mode,
                            [], seeds, data["plens"], device, reference,
                            old_lp=None, zero_grad=False, step=False,
                            loss_scale=prompt_scale,
                        )
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in student.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    update_steps += 1

                elapsed = max(time.monotonic() - started, 1e-6)
                rate = (visits - args.start_visit) / elapsed * 3600
                loss_text = "-" if loss is None else f"{loss:.4f}"
                print(
                    f"  [update] step={update_steps} visit={visits} "
                    f"valid_prompts={len(prepared)}/{len(update_batch)} "
                    f"loss={loss_text} used={used} skipped={skipped} "
                    f"rate={rate:.1f} visits/h", flush=True,
                )

                save_for_update = bool(
                    prepared and args.snap_every_updates
                    and update_steps % args.snap_every_updates == 0
                )
                save_for_visit = bool(
                    args.snap_every and visits % args.snap_every == 0
                )
                if save_for_update or save_for_visit:
                    snapshot = output / f"snap{visits:06d}"
                    save_checkpoint(snapshot)
                    metadata = {
                        "optimizer_update": update_steps,
                        "prompt_visits": visits,
                        "prompt_update_batch_size": args.prompt_update_batch_size,
                        "valid_prompts_in_update": len(prepared),
                        "skipped_prompts_in_batch": batch_skipped,
                    }
                    (snapshot / "update_metadata.json").write_text(
                        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                    )
                    print(
                        f"  [snap] update={update_steps} visits={visits} -> "
                        f"{snapshot}", flush=True,
                    )
            del per_prompt
            torch.cuda.empty_cache()
        if visits >= target_visits:
            break

    save_checkpoint(output)
    print(
        f"[esrl] done. used={used} skipped={skipped}; saved -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
