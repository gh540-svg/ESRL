"""Backbone-specific loading helpers for the text-only experiment protocol.

The benchmark remains text-only, but Gemma 4 ships as a multimodal checkpoint
and therefore must be opened with ``AutoModelForMultimodalLM``.  Keeping this
difference in one module prevents the training and evaluation paths from
silently using different model classes.
"""
from __future__ import annotations


def is_gemma4(model_name: str) -> bool:
    if str(model_name).lower().startswith("google/gemma-4-"):
        return True
    # Full-model checkpoint directories must use the same loader as their base.
    import json
    from pathlib import Path
    config = Path(model_name) / "config.json"
    if config.is_file():
        return str(json.loads(config.read_text(encoding="utf-8")).get("model_type", "")).startswith("gemma4")
    return False


def load_tokenizer(model_name: str, **kwargs):
    # The Gemma 4 processor contains the same text tokenizer used by the
    # conditional-generation model.  AutoTokenizer is sufficient for this
    # benchmark because no image/audio inputs are supplied.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def load_text_generation_model(model_name, **kwargs):
    """Load a model that accepts input_ids and returns causal-LM logits."""
    if is_gemma4(str(model_name)):
        from transformers import AutoModelForMultimodalLM

        # Transformers 5 uses ``dtype`` in its Gemma 4 examples.  Accept the
        # legacy keyword used by the established Qwen/Llama experiment code.
        if "torch_dtype" in kwargs and "dtype" not in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
        kwargs.setdefault("attn_implementation", "sdpa")
        return AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def lora_target_modules(model_name: str):
    modules = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
    if is_gemma4(model_name):
        # Gemma 4 E4B also contains vision and audio towers with similarly named
        # projections.  Adapt only its language branch for a fair text-only run.
        return r".*language_model.*\.(?:" + "|".join(modules) + r")"
    return list(modules)


def es_teacher_parameters(model, model_name: str):
    """Parameters perturbed by ES; exclude unused Gemma modality encoders."""
    if not is_gemma4(model_name):
        return list(model.parameters())
    selected = [
        parameter for name, parameter in model.named_parameters()
        if "language_model" in name
    ]
    if not selected:
        raise RuntimeError("Gemma 4 language-model parameters were not found")
    return selected
