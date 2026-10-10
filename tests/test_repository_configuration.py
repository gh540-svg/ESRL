import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryConfigurationTests(unittest.TestCase):
    def test_only_requested_methods_are_exposed(self):
        launcher = (ROOT / "scripts" / "slurm_two_h200.sh").read_text()
        self.assertIn("esrl|es_direct|sft", launcher)
        for excluded in ("train_ttrl", "train_echo", "train_evol_rl"):
            self.assertNotIn(excluded, launcher.lower())
        trainer_names = sorted(path.name for path in (ROOT / "src").glob("train_*.py"))
        self.assertEqual(
            ["train_es_direct.py", "train_esrl.py", "train_sft.py"],
            trainer_names,
        )

    def test_two_h200_allocation(self):
        launcher = (ROOT / "scripts" / "slurm_two_h200.sh").read_text()
        self.assertRegex(launcher, r"#SBATCH --nodes=2\b")
        self.assertRegex(launcher, r"#SBATCH --ntasks-per-node=1\b")
        self.assertRegex(launcher, r"#SBATCH --gres=gpu:nvidia:1\b")

    def test_token_limits_are_pinned(self):
        launcher = (ROOT / "scripts" / "slurm_two_h200.sh").read_text()
        self.assertIn("--max-new 8196", launcher)
        self.assertIn("--max-model-len 12292", launcher)
        for trainer in ("train_esrl.py", "train_es_direct.py", "train_sft.py"):
            text = (ROOT / "src" / trainer).read_text()
            self.assertRegex(text, r'--max-new"?, type=int, default=8196')
            self.assertRegex(text, r'--max-model-len"?, type=int, default=12292')

    def test_esrl_name_replaces_old_public_name(self):
        public_files = list((ROOT / "src").glob("*.py")) + list(
            (ROOT / "scripts").glob("*.sh")
        )
        old_name = re.compile(r"es[_-]?grpo", re.I)
        offenders = [
            path.name for path in public_files if old_name.search(path.read_text())
        ]
        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
