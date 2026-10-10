"""PEFT adapter construction and vLLM worker helpers used by all trainers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _attention_dimensions(config):
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    return hidden, heads * head_dim, kv_heads * head_dim


def create_initial_peft_adapter(path: Path, model_name: str, rank: int, seed: int):
    """Create a Kaiming-A/zero-B Qwen2/Qwen3/Llama LoRA adapter."""
    from peft import LoraConfig
    from transformers import AutoConfig

    path.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    model_type = str(getattr(config, "model_type", ""))
    if model_type not in ("qwen3", "qwen2", "llama"):
        raise RuntimeError(
            "the ESRL vLLM tensor mapping supports Qwen2/Qwen3/Llama, "
            f"got {model_type!r}"
        )
    hidden, query_width, kv_width = _attention_dimensions(config)
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
        "self_attn.q_proj": (rank, hidden, query_width, rank),
        "self_attn.k_proj": (rank, hidden, kv_width, rank),
        "self_attn.v_proj": (rank, hidden, kv_width, rank),
        "self_attn.o_proj": (rank, query_width, hidden, rank),
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
            bound = 1.0 / math.sqrt(a_cols)
            matrix_a = torch.empty((a_rows, a_cols), dtype=torch.float32)
            matrix_a.uniform_(-bound, bound, generator=generator)
            matrix_b = torch.zeros((b_rows, b_cols), dtype=torch.float32)
            state[f"{prefix}.lora_A.weight"] = matrix_a
            state[f"{prefix}.lora_B.weight"] = matrix_b
    save_file(
        state,
        str(path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    return path


def adapter_scale(adapter_dir: Path):
    config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
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
            "vLLM is missing; install the pinned requirements.txt environment"
        ) from error
    numbers = []
    for piece in raw.split(".")[:2]:
        digits = "".join(character for character in piece if character.isdigit())
        numbers.append(int(digits or 0))
    if tuple(numbers) != (0, 11):
        raise SystemExit(f"ESRL requires vLLM 0.11.x, found {raw}")
    return raw


def rpc_one(engine, method, *args):
    results = engine.collective_rpc(method, args=tuple(args))
    if not results:
        raise RuntimeError(f"vLLM RPC {method} returned no worker result")
    return results[0]
