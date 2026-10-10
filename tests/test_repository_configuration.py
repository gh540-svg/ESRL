import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryConfigurationTests(unittest.TestCase):
    def test_esrl_is_the_only_trainer(self):
        trainer_names = sorted(path.name for path in (ROOT / "src").glob("train_*.py"))
        self.assertEqual(["train_esrl.py"], trainer_names)
        launcher = (ROOT / "scripts" / "slurm_two_h200.sh").read_text()
        self.assertIn("train_esrl.py", launcher)

    def test_single_seed_matrix(self):
        submitter = (ROOT / "scripts" / "submit_matrix.sh").read_text()
        self.assertIn("SEED=1", submitter)
        self.assertNotRegex(submitter, r"for seed in")
        self.assertIn("Submitted 12 ESRL seed-1 jobs", submitter)

    def test_two_h200_allocation(self):
        launcher = (ROOT / "scripts" / "slurm_two_h200.sh").read_text()
        self.assertRegex(launcher, r"#SBATCH --nodes=2\b")
        self.assertRegex(launcher, r"#SBATCH --ntasks-per-node=1\b")
        self.assertRegex(launcher, r"#SBATCH --gres=gpu:nvidia:1\b")

    def test_token_limits_are_pinned(self):
        launcher = (ROOT / "scripts" / "slurm_two_h200.sh").read_text()
        trainer = (ROOT / "src" / "train_esrl.py").read_text()
        self.assertIn("--max-new 8196", launcher)
        self.assertIn("--max-model-len 12292", launcher)
        self.assertRegex(trainer, r'--max-new"?, type=int, default=8196')
        self.assertRegex(trainer, r'--max-model-len"?, type=int, default=12292')

    def test_old_public_name_is_absent(self):
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
