# ESRL

# 📒 Overview

**ESRL** is a  a continual test-time learning method designed to mitigate catastrophic forgetting during label-free continual adaptation. Specifically, ESRL constructs a population of locally perturbed policies around the current model to explore multiple neighboring parameter regions throughout adaptation, and then samples trajectories from those regions. Unlike trajectory-space exploration from a single evolving policy, this parameter-space population maintains diverse sources of self-generated supervision even when the central policy begins to drift. The resulting trajectories are aggregated into label-free group-relative learning signals to update the central student.

---

# 🚀 Quick Start

## 1. Installation

The code targets Linux, Python 3.10, CUDA-capable GPUs, and vLLM 0.11.x.
Because vLLM selects a compatible PyTorch build, install the dependencies in a
dedicated environment:

```bash
git clone https://github.com/gh540-svg/ESRL.git
cd ESRL

conda create -y -n esrl python=3.10
conda activate esrl
python -m pip install -r requirements.txt
```

Download the selected Hugging Face models and datasets before enabling offline
mode. On the cluster used for the paper, the launchers default to:

```bash
export VLLM_PYTHON_ENV="$HOME/conda-envs/esrl-vllm011"
export HF_HOME="$HOME/experiments/ESRL/.hf"
```

Override either path when your environment or cache is elsewhere.

## 2. Code Structure

```text
ESRL/
├── README.md
├── BASELINES.md
├── requirements.txt
├── scripts/
│   ├── slurm_two_h200.sh            # one ESRL job
│   ├── submit_matrix.sh             # 4 models × 3 domains
│   └── slurm_baselines_two_h200.sh  # local ES-Direct / Mode-SFT jobs
├── src/
│   ├── train_esrl.py                # ESRL training loop
│   ├── train_es_direct.py           # ES-Direct baseline
│   ├── train_sft.py                 # label-free Mode-SFT baseline
│   ├── esrl_core.py                 # student construction and updates
│   ├── common.py                    # data, prompts, parsing, and scoring
│   ├── vllm_adapter.py              # PEFT/vLLM adapter utilities
│   ├── vllm_es_worker.py            # perturbation worker
│   ├── remote_vllm_engine.py        # training-node client
│   └── vllm_remote_server.py        # rollout-node service
└── tests/
    └── test_repository_configuration.py
```

## 3. Reproduce ESRL

The published launcher requests two H200 nodes: one H200 runs the PEFT student
and frozen KL reference, while the second runs the persistent vLLM rollout
engine.

Submit one run from the repository root:

```bash
sbatch --job-name=esrl_q4_math_s1 \
  scripts/slurm_two_h200.sh math 4b tok8196_main 1
```

Arguments are `DOMAIN MODEL_KEY [RUN_TAG [SEED]]`.

- Domains: `math`, `knowledge`, `medical`
- Models: `0p6b`, `4b`, `8b`, `llama3p2_3b`
- Default seed: `1`

Submit the complete paper matrix:

```bash
bash scripts/submit_matrix.sh tok8196_main
```

This submits 12 independent jobs: four models × three training domains, all
using ESRL seed 1.

### Published configuration

| Setting | Value |
|---|---:|
| Training generation limit | `8196` new tokens |
| vLLM model length | `12292` tokens |
| Rollout population | `16` |
| Prompt visits per run | `2000` |
| LoRA rank | `16` |
| Perturbation sigma | `0.001` |
| Learning rate | `1e-5` |
| Base-policy KL coefficient | `0.001` |
| Hardware per job | `2 × H200` on two nodes |

The `8196` generation limit is intentional. `max_model_len=12292` reserves
an additional 4096 tokens for the prompt and chat template.

# 🏡 Datasets and Baselines

## Datasets used in the paper

| Domain | Dataset | Adaptation/evaluation split used by the launcher |
|---|---|---|
| Math | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Shuffled test split: 400 adaptation prompts, 100 held-out evaluation prompts; 5 epochs |
| Knowledge | [MMLU](https://huggingface.co/datasets/cais/mmlu) | Shuffled `all` test split: 500 adaptation prompts, 200 held-out evaluation prompts; 4 epochs |
| Medical | [MedQA-USMLE](https://huggingface.co/datasets/mkieffer/MedQA-USMLE) | 500 shuffled train prompts for adaptation and 200 shuffled test prompts for evaluation; 4 epochs |

The launcher fixes `data_seed=0`. For MMLU and MedQA, answer choices are
permuted during adaptation to reduce position shortcuts.

## Baselines

The external reference implementations are:

- [TTRL](https://github.com/PRIME-RL/TTRL)
- [EVOL-RL](https://github.com/YujunZhou/EVOL-RL)
- [ECHO](https://github.com/user683/ECHO)

The repository also includes separate, runnable implementations of
**ES-Direct** and label-free **Mode-SFT**. Their definitions, matched settings,
and launch commands are in [BASELINES.md](BASELINES.md).

# ✅ Validation

The following checks do not require a GPU:

```bash
python -m compileall -q src
python -m unittest discover -s tests -v
bash -n scripts/slurm_two_h200.sh
bash -n scripts/slurm_baselines_two_h200.sh
bash -n scripts/submit_matrix.sh
```

Model weights, caches, checkpoints, generated results, Slurm logs, and runtime
adapter exchange directories are excluded from version control.

# 🎁 Acknowledgments

ESRL builds on open-source work in vLLM, Hugging Face Transformers, PEFT, and
the test-time learning methods linked above. We thank their authors for making
their code and datasets available.
