"""vLLM worker extension for persistent, GPU-resident LoRA ES.

This module deliberately contains no driver-side vLLM imports.  vLLM imports
the extension by its qualified name inside the worker process.  The extension
keeps a float32 central LoRA on the GPU and materialises one independently
perturbed copy into every active vLLM LoRA slot before generation.

The code targets vLLM 0.11.x's V1 worker API. The accompanying Slurm launcher
sets ``VLLM_USE_V1=1``; the V1 model runner exposes its LoRA worker manager at
``self.model_runner.lora_manager``.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import torch


class VLLMESWorkerExtension:
    """Methods exposed through :meth:`vllm.LLM.collective_rpc`.

    vLLM folds PEFT's ``lora_alpha / r`` factor into matrix B when it loads an
    adapter.  ``_es_master`` therefore stores A in PEFT coordinates and B in
    vLLM's scaled coordinates.  Noise and permanent ES updates to B are scaled
    by the same factor.  Export divides B by the factor again.
    """

    def _es_manager(self):
        lora_manager = getattr(self.model_runner, "lora_manager", None)
        if lora_manager is None:
            raise RuntimeError(
                "vLLM LoRA manager is unavailable. Use vLLM 0.11.x with "
                "VLLM_USE_V1=1 and enable_lora=True."
            )
        manager = getattr(lora_manager, "_adapter_manager", None)
        if manager is None:
            raise RuntimeError("vLLM adapter manager is unavailable")
        return manager

    @staticmethod
    def _internal_to_peft(module_key: str, matrix: str) -> str:
        """Map a vLLM Qwen3 module/sub-layer to a PEFT safetensors key."""
        module_name, marker, sub_text = module_key.partition("#sub")
        if marker:
            sub_index = int(sub_text)
            if module_name.endswith(".qkv_proj"):
                names = ("q_proj", "k_proj", "v_proj")
                module_name = module_name[:-len("qkv_proj")] + names[sub_index]
            elif module_name.endswith(".gate_up_proj"):
                names = ("gate_proj", "up_proj")
                module_name = module_name[:-len("gate_up_proj")] + names[sub_index]
            else:
                raise RuntimeError(f"unsupported packed Qwen3 LoRA module {module_key!r}")
        if not module_name.startswith("model."):
            raise RuntimeError(
                f"unexpected Qwen3 LoRA module {module_name!r}; expected model.*"
            )
        return f"base_model.model.{module_name}.lora_{matrix}.weight"

    def _es_gpu_views(self, adapter_id: int):
        """Return cropped active-slot views keyed by vLLM module name.

        Cropping uses the registered CPU LoRA shapes, avoiding rank/output
        padding in vLLM's stacked buffers from becoming trainable parameters.
        """
        manager = self._es_manager()
        try:
            slot = manager.lora_index_to_id.index(int(adapter_id))
        except ValueError as error:
            raise RuntimeError(f"LoRA adapter {adapter_id} is not active") from error

        registered = manager.list_adapters().get(int(adapter_id))
        if registered is None:
            raise RuntimeError(f"LoRA adapter {adapter_id} is not registered")

        views = {}
        for module_name, module in sorted(manager.modules.items()):
            a_stacked = getattr(module, "lora_a_stacked", ())
            b_stacked = getattr(module, "lora_b_stacked", ())
            if not isinstance(a_stacked, (list, tuple)) or not isinstance(
                b_stacked, (list, tuple)
            ):
                continue
            cpu_lora = registered.get_lora(module_name)
            for sub_index, (a_all, b_all) in enumerate(zip(a_stacked, b_stacked)):
                if not isinstance(a_all, torch.Tensor) or not isinstance(
                    b_all, torch.Tensor
                ):
                    continue
                cpu_a = getattr(cpu_lora, "lora_a", None)
                cpu_b = getattr(cpu_lora, "lora_b", None)
                if isinstance(cpu_a, (list, tuple)):
                    cpu_a = cpu_a[sub_index]
                    cpu_b = cpu_b[sub_index]
                if cpu_a is None or cpu_b is None:
                    continue
                rank, input_dim = cpu_a.shape
                output_dim, b_rank = cpu_b.shape
                if rank != b_rank:
                    raise RuntimeError(f"rank mismatch in {module_name}")
                key = (
                    f"{module_name}#sub{sub_index}"
                    if len(a_stacked) > 1
                    else module_name
                )
                views[key] = (
                    a_all[slot, 0, :rank, :input_dim],
                    b_all[slot, 0, :output_dim, :rank],
                )
        if not views:
            raise RuntimeError("no active Qwen3 LoRA tensors were found")
        return views

    @torch.inference_mode()
    def es_initialize(
        self,
        adapter_ids: list[int],
        peft_checkpoint: str,
        lora_scale: float,
    ):
        """Load the exact FP32 PEFT centre and synchronise every GPU slot."""
        from safetensors.torch import load_file

        if not adapter_ids:
            raise ValueError("adapter_ids must not be empty")
        if not math.isfinite(lora_scale) or lora_scale <= 0:
            raise ValueError("lora_scale must be finite and positive")

        manager = self._es_manager()
        for adapter_id in adapter_ids:
            try:
                manager.pin_adapter(int(adapter_id))
            except (AttributeError, NotImplementedError):
                # max_cpu_loras == max_loras also prevents eviction in the
                # non-LRU manager, where explicit pinning is unsupported.
                pass

        state_path = Path(peft_checkpoint) / "adapter_model.safetensors"
        peft_state = load_file(str(state_path), device="cpu")
        first_views = self._es_gpu_views(int(adapter_ids[0]))
        master = {}
        missing = []
        for module_key, (gpu_a, gpu_b) in sorted(first_views.items()):
            a_key = self._internal_to_peft(module_key, "A")
            b_key = self._internal_to_peft(module_key, "B")
            if a_key not in peft_state or b_key not in peft_state:
                missing.extend(key for key in (a_key, b_key) if key not in peft_state)
                continue
            central_a = peft_state[a_key].to(device=gpu_a.device, dtype=torch.float32)
            central_b = peft_state[b_key].to(device=gpu_b.device, dtype=torch.float32)
            if central_a.shape != gpu_a.shape or central_b.shape != gpu_b.shape:
                raise RuntimeError(
                    f"LoRA shape mismatch for {module_key}: checkpoint "
                    f"{tuple(central_a.shape)}/{tuple(central_b.shape)}, vLLM "
                    f"{tuple(gpu_a.shape)}/{tuple(gpu_b.shape)}"
                )
            master[module_key] = (
                central_a.clone(),
                central_b.mul(float(lora_scale)).clone(),
            )
        if missing:
            raise RuntimeError(
                "PEFT checkpoint is missing vLLM LoRA tensors: "
                + ", ".join(missing[:5])
            )

        self._es_adapter_ids = [int(value) for value in adapter_ids]
        self._es_scale = float(lora_scale)
        self._es_master = master
        self._es_sync_slots()
        trainable = sum(a.numel() + b.numel() for a, b in master.values())
        return {
            "adapters": len(adapter_ids),
            "lora_pairs": len(master),
            "trainable_parameters": trainable,
            "master_dtype": "float32",
        }

    def _es_require_initialized(self):
        if not hasattr(self, "_es_master"):
            raise RuntimeError("call es_initialize before ES operations")

    @torch.inference_mode()
    def _es_sync_slots(self):
        self._es_require_initialized()
        for adapter_id in self._es_adapter_ids:
            views = self._es_gpu_views(adapter_id)
            if set(views) != set(self._es_master):
                raise RuntimeError("LoRA tensor layout differs between adapter slots")
            for key, (target_a, target_b) in views.items():
                source_a, source_b = self._es_master[key]
                target_a.copy_(source_a.to(target_a.dtype))
                target_b.copy_(source_b.to(target_b.dtype))

    @torch.inference_mode()
    def es_materialize(self, seeds: list[int], sigma: float):
        """Write theta + sigma*epsilon_k into all active inference slots."""
        self._es_require_initialized()
        if len(seeds) != len(self._es_adapter_ids):
            raise ValueError(
                f"received {len(seeds)} seeds for {len(self._es_adapter_ids)} adapters"
            )
        device = next(iter(self._es_master.values()))[0].device
        for adapter_id, seed in zip(self._es_adapter_ids, seeds):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            views = self._es_gpu_views(adapter_id)
            for key in sorted(self._es_master):
                master_a, master_b = self._es_master[key]
                target_a, target_b = views[key]
                noise_a = torch.randn(
                    master_a.shape, generator=generator, device=device,
                    dtype=torch.float32,
                )
                noise_b = torch.randn(
                    master_b.shape, generator=generator, device=device,
                    dtype=torch.float32,
                )
                target_a.copy_((master_a + float(sigma) * noise_a).to(target_a.dtype))
                target_b.copy_(
                    (master_b + self._es_scale * float(sigma) * noise_b).to(
                        target_b.dtype
                    )
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True

    @torch.inference_mode()
    def es_update(
        self,
        seeds: list[int],
        advantages: list[float],
        alpha: float,
        population: int,
    ):
        """Apply theta += alpha/N * sum advantage_k * epsilon_k in FP32."""
        self._es_require_initialized()
        if len(seeds) != len(advantages):
            raise ValueError("seeds and advantages must have equal length")
        if population <= 0:
            raise ValueError("population must be positive")
        device = next(iter(self._es_master.values()))[0].device
        for seed, advantage in zip(seeds, advantages):
            coefficient = float(alpha) * float(advantage) / int(population)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            for key in sorted(self._es_master):
                master_a, master_b = self._es_master[key]
                noise_a = torch.randn(
                    master_a.shape, generator=generator, device=device,
                    dtype=torch.float32,
                )
                noise_b = torch.randn(
                    master_b.shape, generator=generator, device=device,
                    dtype=torch.float32,
                )
                master_a.add_(noise_a, alpha=coefficient)
                master_b.add_(noise_b, alpha=coefficient * self._es_scale)
        return True

    @torch.inference_mode()
    def _es_export_state(self):
        """Return the FP32 centre in ordinary, unscaled PEFT coordinates."""
        self._es_require_initialized()
        state = {}
        for module_key, (master_a, master_b) in sorted(self._es_master.items()):
            state[self._internal_to_peft(module_key, "A")] = (
                master_a.detach().cpu().contiguous()
            )
            state[self._internal_to_peft(module_key, "B")] = (
                master_b.detach().div(self._es_scale).cpu().contiguous()
            )
        return state

    @torch.inference_mode()
    def es_save_checkpoint(self, output_dir: str, adapter_config_path: str):
        """Atomically export the FP32 centre as an ordinary PEFT adapter."""
        from safetensors.torch import save_file

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        state = self._es_export_state()

        temporary = destination / "adapter_model.safetensors.tmp"
        final = destination / "adapter_model.safetensors"
        save_file(state, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, final)
        shutil.copy2(adapter_config_path, destination / "adapter_config.json")
        metadata = {
            "engine": "vllm-0.11-v1",
            "central_dtype": "float32",
            "population": len(self._es_adapter_ids),
            "lora_scale_folded_during_inference": self._es_scale,
            "parameter_order": "sorted_vllm_qwen3_module_then_A_then_B",
        }
        temporary_json = destination / "vllm_es_state.json.tmp"
        temporary_json.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary_json, destination / "vllm_es_state.json")
        return {"tensors": len(state), "bytes": final.stat().st_size}
