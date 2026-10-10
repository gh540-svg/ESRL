"""ES-Direct LoRA trainer with persistent vLLM multi-LoRA rollouts.

It preserves these experiment semantics:

* N independent parameter perturbations per prompt (default N=16)
* one stochastic rollout from each population member
* modal/majority pseudo-reward and the same confidence gate
* one direct ES update per accepted prompt
* 500-visit PEFT snapshots compatible with the existing evaluator

The rollout H200 runs one persistent vLLM engine holding the Qwen/Llama base,
    the FP32 central LoRA, and all N perturbed LoRA inference slots on one GPU.
    The tensor layout supports Qwen2/Qwen3 and Llama decoder backbones.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

import common as C


TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--dataset", required=True, choices=("math500", "mmlu", "medqa"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--eval-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--rollouts-per-member", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--max-new", type=int, default=8196)
    parser.add_argument("--p-high", type=float, default=0.95)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--snap-every", type=int, default=500)
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--start-visit", type=int, default=0)
    parser.add_argument(
        "--max-visits", type=int, default=None,
        help="stop after exactly this many total prompt visits",
    )
    parser.add_argument("--permute-choices", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-seed", type=int, default=None,
        help="dataset split/permutation seed; defaults to --seed",
    )
    parser.add_argument("--max-model-len", type=int, default=12292)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=None,
        help=(
            "optional vLLM scheduler token budget; omission lets vLLM 0.11 "
            "select its GPU-specific A100/H200 default"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="disable CUDA graphs/compilation as a stability fallback",
    )
    args = parser.parse_args()
    if args.population <= 1:
        parser.error("--population must be at least 2")
    if args.rollouts_per_member != 1:
        parser.error("this direct-ES arm requires exactly one rollout per member")
    if args.lora_r not in (8, 16, 32, 64, 128, 256, 320, 512):
        parser.error("--lora-r is unsupported by vLLM")
    if args.sigma < 0 or args.alpha < 0:
        parser.error("--sigma and --alpha must be non-negative")
    if args.seed < 0 or (args.data_seed is not None and args.data_seed < 0):
        parser.error("--seed and --data-seed must be non-negative")
    if args.max_visits is not None and args.max_visits <= args.start_visit:
        parser.error("--max-visits must be greater than --start-visit")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if (
        args.max_num_batched_tokens is not None
        and args.max_num_batched_tokens < args.max_model_len
    ):
        parser.error("--max-num-batched-tokens must be at least --max-model-len")
    return args


def _attention_dimensions(config):
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    return hidden, heads * head_dim, kv_heads * head_dim


def create_initial_peft_adapter(path: Path, model_name: str, rank: int, seed: int):
    """Create a Kaiming-A/zero-B Qwen2/Qwen3/Llama LoRA without base weights."""
    from peft import LoraConfig
    from transformers import AutoConfig

    path.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    model_type = str(getattr(config, "model_type", ""))
    if model_type not in ("qwen3", "qwen2", "llama"):
        raise RuntimeError(
            "the vLLM ES tensor mapping currently supports Qwen2/Qwen3/Llama, "
            f"got {model_type!r}"
        )
    hidden, q_width, kv_width = _attention_dimensions(config)
    intermediate = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        target_modules=list(TARGET_MODULES),
        task_type="CAUSAL_LM",
        inference_mode=True,
    )
    lora_config.base_model_name_or_path = model_name
    lora_config.save_pretrained(path)

    shapes = {
        "self_attn.q_proj": (rank, hidden, q_width, rank),
        "self_attn.k_proj": (rank, hidden, kv_width, rank),
        "self_attn.v_proj": (rank, hidden, kv_width, rank),
        "self_attn.o_proj": (rank, q_width, hidden, rank),
        "mlp.gate_proj": (rank, hidden, intermediate, rank),
        "mlp.up_proj": (rank, hidden, intermediate, rank),
        "mlp.down_proj": (rank, intermediate, hidden, rank),
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    state = {}
    for layer in range(layers):
        for module_name, (a_rows, a_cols, b_rows, b_cols) in shapes.items():
            prefix = f"base_model.model.model.layers.{layer}.{module_name}"
            # torch.nn.Linear's default LoRA A initialisation is kaiming_uniform_(a=sqrt(5)).
            bound = 1.0 / math.sqrt(a_cols)
            a = torch.empty((a_rows, a_cols), dtype=torch.float32)
            a.uniform_(-bound, bound, generator=generator)
            b = torch.zeros((b_rows, b_cols), dtype=torch.float32)
            state[f"{prefix}.lora_A.weight"] = a
            state[f"{prefix}.lora_B.weight"] = b
    save_file(state, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    return path


def adapter_scale(adapter_dir: Path):
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    rank = float(config["r"])
    alpha = float(config["lora_alpha"])
    return alpha / math.sqrt(rank) if config.get("use_rslora", False) else alpha / rank


def validate_adapter(path: Path):
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise RuntimeError("incomplete PEFT adapter: " + ", ".join(missing))


def vllm_version_check():
    try:
        from importlib.metadata import version
        raw = version("vllm")
    except Exception as error:
        raise SystemExit(
            "vLLM is not installed in this Python environment. Install the pinned "
            "requirements_vllm_es.txt in a separate/compatible environment."
        ) from error
    numbers = []
    for piece in raw.split(".")[:2]:
        digits = "".join(character for character in piece if character.isdigit())
        numbers.append(int(digits or 0))
    if tuple(numbers) != (0, 11):
        raise SystemExit(
            f"this worker integration requires vLLM 0.11.x, found {raw}; "
            "install the pinned requirements.txt environment"
        )
    return raw


def rpc_one(engine, method, *args):
    results = engine.collective_rpc(method, args=tuple(args))
    if not results:
        raise RuntimeError(f"vLLM RPC {method} returned no worker result")
    return results[0]


def save_centre(engine, output_dir: Path, adapter_dir: Path):
    result = rpc_one(
        engine,
        "es_save_checkpoint",
        str(output_dir.resolve()),
        str((adapter_dir / "adapter_config.json").resolve()),
    )
    print(
        f"  [checkpoint] {result['tensors']} tensors, "
        f"{result['bytes'] / 2**20:.1f} MiB -> {output_dir}",
        flush=True,
    )


def build_rollout_prompts(item, population, seeds, permute, tok, instruct):
    if not permute:
        return [item] * population, [None] * population
    built = [
        C.build_mmlu_rollout_prompt(
            item, tok, instruct, random.Random(int(seed) * 7919 + 1)
        )
        for seed in seeds
    ]
    return [value[0] for value in built], [value[1] for value in built]


def main():
    args = parse_args()
    data_seed = args.seed if args.data_seed is None else args.data_seed
    # vLLM 0.11 exposes ``LLMEngine`` through the V1 implementation. Forcing
    # V0 makes EngineArgs select V1 while the environment reports V0, so the
    # engine aborts before loading the model.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    version = vllm_version_check()

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    if os.environ.get("VLLM_REMOTE_HOST"):
        from remote_vllm_engine import RemoteVLLMEngine as LLM
    else:
        from vllm import LLM

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.init_adapter:
        adapter_dir = Path(args.init_adapter).resolve()
    else:
        adapter_dir = output / ".vllm_initial_adapter"
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            create_initial_peft_adapter(
                adapter_dir, args.model, args.lora_r, args.seed
            )
    validate_adapter(adapter_dir)
    scale = adapter_scale(adapter_dir)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    instruct = "Instruct" in args.model
    permute = args.permute_choices and args.dataset in ("mmlu", "medqa")
    train_prompts, _ = C.load_task(
        args.dataset, tok, instruct, args.eval_size, args.train_size,
        data_seed, permute,
    )
    C.set_task(args.dataset)
    expected_visits = len(train_prompts) * args.epochs
    if args.start_visit < 0 or args.start_visit > expected_visits:
        raise SystemExit(
            f"--start-visit must be in [0, {expected_visits}], got {args.start_visit}"
        )
    if args.max_visits is not None and args.max_visits > expected_visits:
        raise SystemExit(
            f"--max-visits={args.max_visits} exceeds the {expected_visits} "
            "visits supplied by --train-size and --epochs"
        )
    target_visits = (
        args.max_visits if args.max_visits is not None else expected_visits
    )

    print(
        f"[es_direct_lora_vllm] vLLM={version} model={args.model} "
        f"population={args.population} LoRA-r={args.lora_r} scale={scale:g}",
        flush=True,
    )
    print(
        f"[es_direct_lora_vllm] {len(train_prompts)} {args.dataset} prompts x "
        f"{args.epochs} epochs = {expected_visits} visits; "
        f"starting at {args.start_visit}; stopping at {target_visits}",
        flush=True,
    )

    engine = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        enable_lora=True,
        max_loras=args.population,
        max_cpu_loras=args.population,
        max_lora_rank=args.lora_r,
        max_model_len=args.max_model_len,
        max_num_seqs=args.population,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
        worker_extension_cls="vllm_es_worker.VLLMESWorkerExtension",
    )
    requests = [
        LoRARequest(f"es_member_{member}", member + 1, str(adapter_dir))
        for member in range(args.population)
    ]
    warmup_params = [
        SamplingParams(temperature=0.0, max_tokens=1, seed=args.seed + member)
        for member in range(args.population)
    ]
    engine.generate(
        ["Warm up."] * args.population,
        warmup_params,
        lora_request=requests,
        use_tqdm=False,
    )
    sanity = rpc_one(
        engine,
        "es_initialize",
        [request.lora_int_id for request in requests],
        str(adapter_dir),
        scale,
    )
    print(f"[es_direct_lora_vllm] GPU state: {sanity}", flush=True)

    visits = args.start_visit
    used = skipped = 0
    started = time.monotonic()
    materialize_seconds = generation_seconds = update_seconds = 0.0
    generated_tokens = length_limited = 0
    for epoch in range(args.epochs):
        for local_index, item in enumerate(train_prompts):
            if visits >= target_visits:
                break
            global_index = epoch * len(train_prompts) + local_index
            if global_index < args.start_visit:
                continue
            visits += 1
            seeds = [10_000 * global_index + member for member in range(args.population)]
            phase_started = time.monotonic()
            rpc_one(engine, "es_materialize", seeds, args.sigma)
            materialize_seconds += time.monotonic() - phase_started
            prompts, permutations = build_rollout_prompts(
                item, args.population, seeds, permute, tok, instruct
            )
            sampling = [
                SamplingParams(
                    temperature=args.temp,
                    top_p=0.95,
                    max_tokens=args.max_new,
                    seed=args.seed * 1_000_000 + global_index * args.population + member,
                )
                for member in range(args.population)
            ]
            phase_started = time.monotonic()
            generated = engine.generate(
                prompts,
                sampling,
                lora_request=requests,
                use_tqdm=False,
            )
            generation_seconds += time.monotonic() - phase_started
            for request_output in generated:
                completion = request_output.outputs[0]
                generated_tokens += len(completion.token_ids)
                if completion.finish_reason == "length":
                    length_limited += 1
            texts = [request_output.outputs[0].text for request_output in generated]
            answers = []
            for text, permutation in zip(texts, permutations):
                answer = C.extract_answer(text)
                answers.append(
                    C.canonical_letter(answer, permutation)
                    if permutation is not None
                    else answer
                )

            advantages, mode, confidence, valid = C.group_advantage(answers)
            if not valid or confidence > args.p_high:
                skipped += 1
                outcome = "skip"
            else:
                phase_started = time.monotonic()
                rpc_one(
                    engine,
                    "es_update",
                    seeds,
                    [float(value) for value in advantages.tolist()],
                    args.alpha,
                    args.population,
                )
                update_seconds += time.monotonic() - phase_started
                used += 1
                outcome = "update"

            if global_index % 10 == 0 or visits == target_visits:
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = (visits - args.start_visit) / elapsed * 3600
                completed_visits = visits - args.start_visit
                completed_rollouts = completed_visits * args.population
                print(
                    f"  [progress] visit={visits} {outcome} p={confidence:.2f} "
                    f"mode={mode} used={used} skipped={skipped} "
                    f"rate={rate:.1f} visits/h avg_tokens="
                    f"{generated_tokens / max(completed_rollouts, 1):.1f} "
                    f"length_limit={length_limited / max(completed_rollouts, 1):.1%} "
                    f"time_s(materialize/generate/update)="
                    f"{materialize_seconds:.1f}/{generation_seconds:.1f}/{update_seconds:.1f}",
                    flush=True,
                )
            if args.snap_every and visits % args.snap_every == 0:
                snapshot = output / f"snap{visits:06d}"
                save_centre(engine, snapshot, adapter_dir)
        if visits >= target_visits:
            break

    save_centre(engine, output, adapter_dir)
    print(
        f"[es_direct_lora_vllm] done. used={used} skipped={skipped}; "
        f"saved -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
