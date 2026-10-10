"""CPU-only structural tests for the independent-domain experiment."""
from pathlib import Path
from unittest import mock
import unittest

import run_continual_baselines as CL
import run_indomain_ood as IO

try:
    import torch
    import common as C
except ImportError:  # local lightweight validation; Wolffe environment has torch
    torch = None
    C = None


class ProtocolTests(unittest.TestCase):
    def test_scale_backbones_are_base_models(self):
        self.assertEqual(CL.MODELS["0p6b"], "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(CL.MODELS["8b"], "Qwen/Qwen3-8B-Base")

    def test_job_matrix_has_one_base_and_eighteen_adaptation_arms(self):
        self.assertEqual(1 + len(IO.BASELINE_METHODS) * len(IO.DOMAINS), 19)

    def test_esgrpo_variant_matrix_has_nine_arms(self):
        methods = ("es_grpo_filter", "es_grpo", "es_grpo_filter_n32")
        self.assertEqual(len(methods) * len(IO.DOMAINS), 9)

    def test_requested_repetition_matrix_has_fifteen_jobs(self):
        filtered_repetitions = 3 * len(IO.DOMAINS)
        comparison_arms = 2 * len(IO.DOMAINS)
        self.assertEqual(filtered_repetitions + comparison_arms, 15)

    def test_seed_override_reaches_training_command(self):
        with mock.patch.object(CL, "SEED", 3):
            command = CL.command_for(
                "es_grpo_ff_n16", CL.MODELS["4b"], IO.DOMAINS["math"],
                Path("out"), None, 0, 500,
            )
        self.assertEqual(command[command.index("--seed") + 1], "3")

    def test_each_domain_has_exactly_2000_prompt_visits(self):
        for stage in IO.DOMAINS.values():
            self.assertEqual(stage["train_size"] * stage["epochs"], CL.VISITS)

    def test_all_adaptive_commands_use_sixteen_rollouts(self):
        expected_internal = {
            "ttrl": ("train.py", "grpo_real"),
            "es_direct": ("train.py", "es_direct"),
            "sft_mode": ("train.py", "sft_good"),
            "echo": ("train.py", "echo"),
            "evol_rl": ("train_evol_rl.py", None),
            "es_direct_lora_vllm": (
                "train_es_direct_lora_vllm.py", None
            ),
            "es_grpo": ("train.py", "es_grpo"),
            "es_grpo_filter": ("train.py", "es_grpo"),
            "es_grpo_filter_n32": ("train.py", "es_grpo"),
        }
        for public_method in IO.PUBLIC_METHODS:
            internal = IO.INTERNAL_METHOD.get(public_method, public_method)
            for stage in IO.DOMAINS.values():
                command = CL.command_for(
                    internal,
                    CL.MODELS["4b"],
                    stage,
                    Path("out"),
                    source=None,
                    current=0,
                    snap_every=500,
                )
                expected_population = (
                    "32" if public_method == "es_grpo_filter_n32" else "16"
                )
                self.assertEqual(
                    command[command.index("--population") + 1],
                    expected_population,
                )
                self.assertNotIn("--init-adapter", command)
                executable, trainer_method = expected_internal[public_method]
                self.assertEqual(command[2], executable)
                if trainer_method is not None:
                    self.assertEqual(command[command.index("--method") + 1], trainer_method)

    def test_echo_released_fixed_budget_defaults_are_explicit(self):
        command = CL.command_for(
            "echo", CL.MODELS["4b"], IO.DOMAINS["math"], Path("out"),
            None, 0, 500,
        )
        expected = {
            "--echo-alpha": "1.0",
            "--echo-beta": "1.0",
            "--echo-s-max": "2.0",
            "--echo-mult-coeff": "0.1",
            "--echo-tail-window": "16",
            "--echo-confidence-topk": "5",
            "--echo-eps-min": "0.05",
            "--echo-eps-max": "0.2",
            "--echo-kappa": "4.0",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_es_grpo_arm_generates_sixteen_and_filters_to_half(self):
        command = CL.command_for(
            "es_grpo_ff_n16", CL.MODELS["4b"], IO.DOMAINS["math"],
            Path("out"), None, 0, 500,
        )
        self.assertEqual(command[command.index("--population") + 1], "16")
        self.assertEqual(command[command.index("--freq-filter") + 1], "0.5")
        self.assertEqual(command[command.index("--method") + 1], "es_grpo")

    def test_es_grpo_n32_arm_generates_thirty_two_and_filters_to_half(self):
        internal = IO.INTERNAL_METHOD["es_grpo_filter_n32"]
        command = CL.command_for(
            internal, CL.MODELS["4b"], IO.DOMAINS["math"],
            Path("out"), None, 0, 500,
        )
        self.assertEqual(command[command.index("--population") + 1], "32")
        self.assertEqual(command[command.index("--freq-filter") + 1], "0.5")
        self.assertEqual(
            IO.arm_name("es_grpo_filter_n32", "4b", "math", "repeat2"),
            "es_grpo_filter_n32_qwen3_4b_train_math_n32_repeat2",
        )

    def test_indomain_vllm_arm_uses_vllm_trainer_and_n16(self):
        command = CL.command_for(
            "es_direct_lora_vllm", CL.MODELS["4b"], IO.DOMAINS["math"],
            Path("out"), None, 0, 500,
        )
        self.assertEqual(command[2], "train_es_direct_lora_vllm.py")
        self.assertEqual(command[command.index("--population") + 1], "16")
        self.assertEqual(
            IO.arm_name(
                "es_direct_lora_vllm", "4b", "math", "a100run"
            ),
            "es_direct_lora_vllm_qwen3_4b_train_math_n16_a100run",
        )

    def test_h200_prompt_batch_applies_only_to_frozen_policy_methods(self):
        with mock.patch.dict(
            "os.environ", {"QWEN3_ROLLOUT_PROMPT_BATCH_SIZE": "4"}
        ):
            for method in ("ttrl", "echo", "evol_rl"):
                command = CL.command_for(
                    method, CL.MODELS["4b"], IO.DOMAINS["math"],
                    Path("out"), None, 0, 500,
                )
                self.assertEqual(
                    command[command.index("--rollout-prompt-batch-size") + 1],
                    "4",
                )
            for method in ("mode_sft", "es_grpo_ff_n16"):
                command = CL.command_for(
                    method, CL.MODELS["4b"], IO.DOMAINS["math"],
                    Path("out"), None, 0, 500,
                )
                self.assertNotIn("--rollout-prompt-batch-size", command)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this local Python")
    def test_echo_shaping_preserves_token_lengths_and_clip_bounds(self):
        entropy = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        confidence = [torch.tensor([0.9, 0.8]), torch.tensor([0.2])]
        shaped, epsilon = C.echo_hybrid_advantages(
            torch.tensor([1.0, -1.0]), entropy, confidence
        )
        self.assertEqual([value.numel() for value in shaped], [2, 1])
        self.assertEqual(len(epsilon), 2)
        self.assertTrue(all(0.05 <= float(value) <= 0.2 for value in epsilon))


if __name__ == "__main__":
    unittest.main()
