# ESRL

Focused research implementation of **ESRL**, with two comparison methods:
**ES-Direct** and **SFT**. The repository intentionally excludes the TTRL,
ECHO, and EVOL-RL trainers.

The `sft` arm is the label-free Mode-SFT baseline used in the experiment
matrix: it generates 16 policy responses, takes the unique modal answer as a
pseudo-label, and applies supervised loss only to responses matching that
mode. It is not gold-label SFT.

## Published configuration

| Setting | Value |
|---|---:|
| Training generation limit | `8196` new tokens |
| vLLM model length | `12292` tokens |
| Rollout population | `16` |
| Prompt visits per arm | `2000` |
| Hardware per job | `2 × H200` on two nodes |
| Training H200 | PEFT student, optimizer, and optional KL reference |
| Rollout H200 | Persistent vLLM engine and LoRA rollout adapters |

The value `8196` is intentional. `max_model_len=12292` reserves an additional
4096 tokens for the prompt and chat template.

## Methods

- `esrl`: rank-16 LoRA policy, 16 independently perturbed rollout teachers,
  group-relative policy-gradient update, `sigma=0.001`, learning rate `1e-5`,
  and base-policy KL coefficient `0.001`.
- `es_direct`: direct evolution-strategy update over rank-16 LoRA parameters,
  `sigma=0.001` and `alpha=0.0005`; no backpropagation through the update.
- `sft`: Mode-SFT over modal-answer rollouts, learning rate `5e-5`.

Every method uses the same population, prompts, data split, visit budget, and
generation limit. Math uses MATH-500; knowledge uses MMLU; medical uses MedQA.

## Repository layout

```text
src/train_esrl.py          ESRL implementation
src/train_es_direct.py     ES-Direct implementation
src/train_sft.py           Mode-SFT implementation
src/common.py              shared data and scoring utilities
src/esrl_core.py           LoRA student and update routines
src/vllm_es_worker.py      vLLM LoRA perturbation worker
src/remote_vllm_engine.py  training-node client
src/vllm_remote_server.py  rollout-node service
scripts/slurm_two_h200.sh  two-H200 job launcher
scripts/submit_matrix.sh   full experiment matrix submission
```

## Environment

The code targets Python 3.10 and vLLM 0.11.x. Install it in a dedicated
environment because vLLM pins compatible PyTorch versions:

```bash
module purge
module load anaconda/conda3

conda create -y -p "$HOME/conda-envs/esrl-vllm011" python=3.10
export PATH="$HOME/conda-envs/esrl-vllm011/bin:$PATH"
python -m pip install -r requirements.txt
```

Cache the selected Hugging Face models and datasets before enabling offline
mode. The Slurm launcher defaults to:

```bash
export VLLM_PYTHON_ENV="$HOME/conda-envs/esrl-vllm011"
export HF_HOME="$HOME/experiments/ESRL/.hf"
```

Override either variable if the cache or environment is elsewhere.

## Submit one job

Run from the repository root on a Slurm login node:

```bash
export VLLM_PYTHON_ENV="$HOME/conda-envs/esrl-vllm011"
export HF_HOME="$HOME/experiments/ESRL/.hf"

sbatch --job-name=esrl_q4_math_s1 \
  scripts/slurm_two_h200.sh esrl math 4b tok8196_main 1
```

Arguments are `METHOD DOMAIN MODEL_KEY [RUN_TAG [SEED]]`.

- method: `esrl`, `es_direct`, `sft`
- domain: `math`, `knowledge`, `medical`
- model: `0p6b`, `4b`, `8b`, `llama3p2_3b`

Examples for the comparison methods:

```bash
sbatch --job-name=esdirect_q4_math \
  scripts/slurm_two_h200.sh es_direct math 4b tok8196_main 0

sbatch --job-name=sft_q4_math \
  scripts/slurm_two_h200.sh sft math 4b tok8196_main 0
```

## Submit the full matrix

```bash
bash scripts/submit_matrix.sh tok8196_main
```

This submits 60 jobs: four models, three training domains, ESRL seeds 1/2/3,
ES-Direct seed 0, and SFT seed 0. The receipt records `8196`, `12292`,
`2xH200`, and every job ID.

## Validation

These checks require no GPU:

```bash
python -m compileall -q src
python -m unittest discover -s tests -v
bash -n scripts/slurm_two_h200.sh
bash -n scripts/submit_matrix.sh
```

Model weights, caches, checkpoints, result files, Slurm logs, and runtime
adapter exchange directories are excluded from version control.
