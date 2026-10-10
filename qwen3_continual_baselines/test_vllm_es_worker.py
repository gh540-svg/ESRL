"""CPU unit tests for the vLLM LoRA ES worker's state/update algebra."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_es_worker import VLLMESWorkerExtension


class _FakeLoRA:
    def __init__(self, a_shape, b_shape):
        self.lora_a = torch.empty(a_shape)
        self.lora_b = torch.empty(b_shape)


class _FakeRegistered:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_lora(self, name):
        return self.mapping.get(name)


class _FakeModule:
    def __init__(self, slots, a_shape, b_shape):
        self.lora_a_stacked = (
            torch.zeros((slots, 1, *a_shape), dtype=torch.bfloat16),
        )
        self.lora_b_stacked = (
            torch.zeros((slots, 1, *b_shape), dtype=torch.bfloat16),
        )


class _FakeManager:
    def __init__(self):
        module_name = "model.layers.0.self_attn.o_proj"
        a_shape, b_shape = (2, 3), (4, 2)
        self.modules = {module_name: _FakeModule(2, a_shape, b_shape)}
        registered = _FakeRegistered({module_name: _FakeLoRA(a_shape, b_shape)})
        self._registered = {1: registered, 2: registered}
        self.lora_index_to_id = [1, 2]

    def list_adapters(self):
        return self._registered

    def pin_adapter(self, _adapter_id):
        return True


class _FakeWorker(VLLMESWorkerExtension):
    def __init__(self):
        manager = _FakeManager()
        self.model_runner = SimpleNamespace(
            lora_manager=SimpleNamespace(_adapter_manager=manager)
        )


class VLLMESWorkerTest(unittest.TestCase):
    def test_materialize_update_and_peft_export(self):
        a_key = (
            "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"
        )
        b_key = (
            "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"
        )
        central_a = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
        central_b = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 20
        worker = _FakeWorker()
        internal_key = "model.layers.0.self_attn.o_proj"
        with patch(
            "safetensors.torch.load_file",
            return_value={a_key: central_a, b_key: central_b},
        ):
            info = worker.es_initialize([1, 2], "unused", 2.0)
        self.assertEqual(info["adapters"], 2)
        self.assertEqual(info["lora_pairs"], 1)
        self.assertEqual(info["trainable_parameters"], 14)

        seeds = [7, 11]
        sigma = 0.1
        worker.es_materialize(seeds, sigma)
        views = worker._es_gpu_views(1)
        slot_a, slot_b = views[internal_key]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seeds[0])
        noise_a = torch.randn(central_a.shape, generator=generator)
        noise_b = torch.randn(central_b.shape, generator=generator)
        torch.testing.assert_close(
            slot_a.float(),
            (central_a + sigma * noise_a).to(torch.bfloat16).float(),
        )
        torch.testing.assert_close(
            slot_b.float(),
            (2 * central_b + 2 * sigma * noise_b).to(torch.bfloat16).float(),
        )

        advantages = [1.5, -0.5]
        alpha = 0.02
        expected_a = central_a.clone()
        expected_b = central_b.clone()
        for seed, advantage in zip(seeds, advantages):
            generator.manual_seed(seed)
            eps_a = torch.randn(central_a.shape, generator=generator)
            eps_b = torch.randn(central_b.shape, generator=generator)
            coefficient = alpha * advantage / 2
            expected_a.add_(eps_a, alpha=coefficient)
            expected_b.add_(eps_b, alpha=coefficient)
        worker.es_update(seeds, advantages, alpha, 2)

        saved = worker._es_export_state()
        torch.testing.assert_close(saved[a_key], expected_a)
        torch.testing.assert_close(saved[b_key], expected_b)

    def test_qwen3_packed_module_names_map_to_peft(self):
        cases = {
            "model.layers.2.self_attn.qkv_proj#sub0": "q_proj",
            "model.layers.2.self_attn.qkv_proj#sub1": "k_proj",
            "model.layers.2.self_attn.qkv_proj#sub2": "v_proj",
            "model.layers.2.mlp.gate_up_proj#sub0": "gate_proj",
            "model.layers.2.mlp.gate_up_proj#sub1": "up_proj",
        }
        for internal, peft_module in cases.items():
            key = VLLMESWorkerExtension._internal_to_peft(internal, "A")
            self.assertEqual(
                key,
                f"base_model.model.model.layers.2."
                f"{'self_attn' if peft_module.endswith('_proj') and peft_module[0] in 'qkv' else 'mlp'}."
                f"{peft_module}.lora_A.weight",
            )


if __name__ == "__main__":
    unittest.main()
