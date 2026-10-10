# Baselines

This file separates the comparison methods from the main ESRL reproduction
path. The ESRL matrix remains in `scripts/submit_matrix.sh`; baseline runs use
their own launcher so the paper configuration cannot be changed accidentally.

## External reference implementations

| Method | Upstream code | Role in the comparison |
|---|---|---|
| TTRL | [PRIME-RL/TTRL](https://github.com/PRIME-RL/TTRL) | Test-time RL with majority-vote pseudo-rewards |
| EVOL-RL | [YujunZhou/EVOL-RL](https://github.com/YujunZhou/EVOL-RL) | Label-free evolutionary RL baseline |
| ECHO | [user683/ECHO](https://github.com/user683/ECHO) | Entropy-confidence hybrid test-time RL |
| ES at Scale | [VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper) | Reference implementation for direct evolution-strategy optimization |

These projects are linked rather than vendored. Follow each upstream
repository for its own installation, license, and reproduction instructions.

## Local ES-Direct baseline

Implementation: [`src/train_es_direct.py`](src/train_es_direct.py)

ES-Direct samples 16 LoRA perturbations for every prompt, generates one rollout
per population member, scores the population with the same modal pseudo-reward
and confidence gate used by ESRL, and applies the reward-weighted ES parameter
update directly. It does not backpropagate through the update.

Pinned settings:

| Setting | Value |
|---|---:|
| Population | `16` |
| Sigma | `0.001` |
| ES step size (`alpha`) | `0.0005` |
| LoRA rank | `16` |
| Prompt visits | `2000` |
| Generation/model length | `8196 / 12292` |

Example:

```bash
sbatch --job-name=esdirect_q4_math \
  scripts/slurm_baselines_two_h200.sh \
  es_direct math 4b tok8196_baselines 0
```

## Local Mode-SFT baseline

Implementation: [`src/train_sft.py`](src/train_sft.py)

Mode-SFT is label-free during adaptation. For each prompt, the current policy
generates 16 responses. When their extracted answers have a unique,
non-unanimous mode that passes the confidence gate, the student receives
supervised loss only on rollouts matching that modal pseudo-label. Gold
evaluation labels are never used for the update.

Pinned settings:

| Setting | Value |
|---|---:|
| Rollouts per prompt | `16` |
| Learning rate | `5e-5` |
| LoRA rank | `16` |
| Prompt visits | `2000` |
| Generation/model length | `8196 / 12292` |

Example:

```bash
sbatch --job-name=sft_q4_math \
  scripts/slurm_baselines_two_h200.sh \
  sft math 4b tok8196_baselines 0
```

## Matched comparison protocol

The local baseline launcher uses the same four model keys, three datasets,
data seed, population, visit budget, maximum generation length, vLLM context
length, and two-H200 allocation as the ESRL launcher. Only the method-specific
optimizer and learning parameters differ.

| Launcher domain | Dataset | Adaptation prompts × epochs | Evaluation prompts |
|---|---|---:|---:|
| `math` | MATH-500 | `400 × 5` | `100` |
| `knowledge` | MMLU | `500 × 4` | `200` |
| `medical` | MedQA-USMLE | `500 × 4` | `200` |
