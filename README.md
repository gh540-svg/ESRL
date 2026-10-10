# ESRL

Focused research implementation of **ESRL**. This repository contains only the
ESRL method and its required runtime utilities; comparison-method trainers are
not included.

ESRL perturbs a rank-16 LoRA policy into 16 rollout teachers, generates one
response from each teacher with persistent vLLM, and applies a group-relative
policy-gradient update to the differentiable student.

## Fixed configuration

| Setting | Value |
|---|---:|
| Experiment seed | `1` |
| Training generation limit | `8196` new tokens |
| vLLM model length | `12292` tokens |
| Rollout population | `16` |
| Prompt visits per arm | `2000` |
| LoRA rank | `16` |
| Perturbation sigma | `0.001` |
| Learning rate | `1e-5` |
| Base-policy KL coefficient | `0.001` |
| Hardware per job | `2 × H200` on two nodes |

The value `8196` is intentional. `max_model_len=12292` reserves an additional
4096 tokens for the prompt and chat template. One H200 runs the PEFT student
and frozen KL reference; the second runs the persistent vLLM rollout engine.

## Repository layout

```text
src/train_esrl.py          ESRL training implementation
src/esrl_core.py           LoRA student and group-relative update
src/common.py              data, prompting, parsing, and scoring utilities
src/vllm_adapter.py        PEFT/vLLM adapter utilities
src/vllm_es_worker.py      vLLM perturbation worker
src/remote_vllm_engine.py  training-node client
src/vllm_remote_server.py  rollout-node service
scripts/slurm_two_h200.sh  one two-H200 ESRL job
scripts/submit_matrix.sh   four-model × three-domain submission
```

## Environment

The code targets Python 3.10 and vLLM 0.11.x:

```bash
module purge
module load anaconda/conda3

conda create -y -p "$HOME/conda-envs/esrl-vllm011" python=3.10
export PATH="$HOME/conda-envs/esrl-vllm011/bin:$PATH"
python -m pip install -r requirements.txt
```

Cache the selected Hugging Face models and datasets before enabling offline
mode. The launcher defaults to:

```bash
export VLLM_PYTHON_ENV="$HOME/conda-envs/esrl-vllm011"
export HF_HOME="$HOME/experiments/ESRL/.hf"
```

## Submit one ESRL job

Arguments are `DOMAIN MODEL_KEY [RUN_TAG [SEED]]`. Seed defaults to `1`.

```bash
sbatch --job-name=esrl_q4_math_s1 \
  scripts/slurm_two_h200.sh math 4b tok8196_main 1
```

- domain: `math`, `knowledge`, `medical`
- model: `0p6b`, `4b`, `8b`, `llama3p2_3b`

## Submit the complete ESRL matrix

```bash
bash scripts/submit_matrix.sh tok8196_main
```

This submits 12 jobs: four models × three independent training domains, all
with ESRL seed `1`, two H200s, `max_new_tokens=8196`, and
`max_model_len=12292`.

## Validation

```bash
python -m compileall -q src
python -m unittest discover -s tests -v
bash -n scripts/slurm_two_h200.sh
bash -n scripts/submit_matrix.sh
```

Model weights, caches, checkpoints, generated results, Slurm logs, and runtime
adapter exchange directories are excluded from version control.
