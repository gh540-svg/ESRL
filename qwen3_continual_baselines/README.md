# Qwen3 continual-learning baselines

This package runs the agreed leakage-safe protocol in the fixed order:

1. MATH-500: 400 adaptation examples x 5 epochs = 2,000 visits; 100 held out.
2. MMLU `all/test`: 500 adaptation examples x 4 epochs = 2,000 visits;
   200 held out. The `auxiliary_train` split is deliberately excluded because
   it contains ARC-derived questions.
3. MedQA-USMLE `train`: 500 adaptation examples x 4 epochs = 2,000 visits;
   evaluation uses the native `test` split.

Seed 0 fixes all splits. The first baseline pass uses rollout population 16.

## Methods

| CLI name | Meaning |
|---|---|
| `es_direct` | Full-parameter evolution strategy, modal-answer reward, no gradient or GRPO second step |
| `es_direct_lora` | Evolution strategy over rank-16 LoRA tensors only, with the same N=16 modal-answer reward and no backpropagation |
| `es_direct_lora_batch16` | Same N=16 LoRA direct ES formula, but materialize 16 perturbed adapters over one shared base and generate all 16 rollouts simultaneously on one H200 |
| `es_direct_lora_vllm` | Accelerated form of the preceding arm: a persistent vLLM 0.11 engine keeps the FP32 LoRA centre and all 16 perturbed inference adapters resident on one H200, then exports normal PEFT snapshots |
| `ttrl` | On-policy clipped GRPO with majority-vote pseudo-reward |
| `sft` | Gold-label supervised fine-tuning with completion-only loss |
| `mode_sft` | Generate 16 student rollouts, select only rollouts matching the unique modal answer, and apply positive-only SFT loss (including all 16 when unanimous) |
| `echo` | Fixed-budget ECHO: 16 flat rollouts with entropy-confidence hybrid advantage shaping and confidence-adaptive PPO clipping |
| `evol_rl` | TTRL update with the upstream EVOL-RL within-band semantic-novelty reward |
| `es_grpo` | Our ES-perturbed-teacher rollout method followed by the group-relative policy-gradient update |
| `es_grpo_p8_r2` | ES-GRPO ablation with 8 perturbed teachers and 2 stochastic rollouts from each teacher (16 total rollouts per prompt); all other ES-GRPO settings are unchanged |
| `es_grpo_ff_p8_r2` | Filtered P8×R2 ES-GRPO: generate 16 rollouts from 8 perturbed teachers, retain the frequency-ranked half (8), then update |
| `es_grpo_ff_n16` | ES-GRPO ablation: generate N=16, retain the frequency-ranked half (8), then update |
| `es_grpo_ff_n32` | ES-GRPO ablation: generate N=32, retain the frequency-ranked half (16), then update |
| `es_grpo_ff_n16_tok1024` | ES-GRPO ablation: generate N=16 with up to 1,024 new tokens, retain the frequency-ranked half (8), then update |

Every method uses the same ordered questions. Mode-SFT, TTRL, ECHO, EVOL-RL,
ES-GRPO and ES-direct never receive adaptation labels. Only SFT receives the
gold adaptation targets. Mode-SFT ignores non-modal rollouts; unlike ES-GRPO,
it does not apply a negative advantage to them.

## Evaluation schedule

Snapshots are saved every 500 visits. At every snapshot the evaluator measures
held-out MATH-500, held-out MMLU, and MedQA test. At the base model and each
stage boundary it additionally measures AMC23, AIME25, ARC-Challenge, GPQA
Diamond, MedMCQA validation, and PubMedQA labeled. This records forgetting at
all checkpoints without making the transfer suite dominate GPU time.

## 1. Upload from Windows PowerShell

Run this on your Windows computer, not inside the SSH terminal:

```powershell
scp -r "$env:USERPROFILE\Documents\New project\qwen3_continual_baselines" `
  "22173342@wolffe.cdms.westernsydney.edu.au:/bigdata/users/22173342/experiments/EvoStrategy/"
```

The destination has no spaces, avoiding the earlier `scp: ambiguous target`
problem.

## 2. Prepare and cache on Wolffe

```bash
cd "$HOME/experiments/EvoStrategy/qwen3_continual_baselines"

module purge
module load anaconda/conda3
export PATH="$HOME/conda-envs/esdgrpo310/bin:$PATH"
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL

python -c 'import transformers; print(transformers.__version__)'
```

If the printed Transformers version is below 4.51, upgrade on a node with
internet access:

```bash
python -m pip install -U -r requirements_qwen3.txt
```

Accept the official GPQA access conditions in your Hugging Face account, then:

```bash
hf auth login

export HF_HOME="$HOME/experiments/EvoStrategy/ES DGRPO/.hf"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
python -u cache_qwen3_assets.py 4b
```

Validate everything using only the cache:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u validate_assets.py 4b
```

Do not submit a GPU job until the last command prints:

```text
[ok] all cached assets and dataset schemas passed
```

## 3. Submit the seven Qwen3-4B jobs

```bash
cd "$HOME/experiments/EvoStrategy/qwen3_continual_baselines"
bash submit_qwen3_baselines.sh 4b
```

The helper prints the seven job IDs. Check them together using its printed
`squeue` command, or use:

```bash
squeue -u "$USER" -o "%.18i %.12P %.24j %.2t %.12M %.15N %R"
```

Follow one job by replacing `JOBID` and `JOBNAME`:

```bash
tail -f "slurm_JOBNAME_JOBID.out"
```

Stop following with `Ctrl-C`; this does not cancel the Slurm job.

### Persistent vLLM ES-Direct LoRA arm

This optional arm removes the repeated Transformers generation overhead from
`es_direct_lora_batch16`. It still uses N=16, one rollout per population
member, modal pseudo-reward, `sigma=0.001`, `alpha=0.0005`, the same confidence
gate, the same stage order, and one update per prompt. Only the rollout engine
and its BF16 inference copies differ; the central adapter is updated and saved
in FP32.

vLLM pins its own compatible PyTorch build, so install it in a dedicated cloned
environment rather than modifying an environment used by running jobs:

```bash
module purge
module load anaconda/conda3
conda create -y -p "$HOME/conda-envs/esdgrpo-vllm011" --clone "$HOME/conda-envs/esdgrpo310"
export PATH="$HOME/conda-envs/esdgrpo-vllm011/bin:$PATH"
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
python -m pip install -r requirements_vllm_es.txt
python -c 'import vllm; print(vllm.__version__)'
```

The last command must print a `0.11.x` version. Submit one H200 job (it is safe
for Slurm to leave it pending until an H200 becomes available):

```bash
cd "$HOME/experiments/EvoStrategy/qwen3_continual_baselines"
export VLLM_PYTHON_ENV="$HOME/conda-envs/esdgrpo-vllm011"
sbatch --job-name=q3_esdl_vllm_4b \
  --export=ALL,VLLM_PYTHON_ENV="$VLLM_PYTHON_ENV" \
  slurm_qwen3_cl_h200.sh es_direct_lora_vllm 4b
```

Its arm/result names are
`es_direct_lora_vllm_qwen3_4b_n16` and
`results_es_direct_lora_vllm_qwen3_4b_n16.json`. Check progress with:

```bash
squeue -u "$USER" -o "%.18i %.12P %.28j %.2t %.12M %.15N %R"
tail -f slurm_q3_esdl_vllm_4b_JOBID.out
```

## 4. Run Qwen3-8B after 4B validation

Cache and validate the 8B backbone, then submit the seven 8B arms:

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
python -u cache_qwen3_assets.py 8b

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u validate_assets.py 8b
bash submit_qwen3_baselines.sh 8b
```

Each completed arm writes a `manifest_*.json` and `results_*.json` in this
directory. Training and evaluation are restartable from their latest complete
snapshot.

Compare the latest completed checkpoint from every available arm with:

```bash
python show_results.py
```

The two frequency-filter ablations are intentionally excluded from the default
baseline submission helper. Submit them explicitly with
`slurm_qwen3_cl_h200.sh es_grpo_ff_n16 4b` and
`slurm_qwen3_cl_h200.sh es_grpo_ff_n32 4b`. The long-rollout ablation is
submitted explicitly with
`slurm_qwen3_cl_h200.sh es_grpo_ff_n16_tok1024 4b`.

Submit the population-8/two-rollouts-per-member ablation explicitly with:

```bash
sbatch --job-name=q3_esgrpo_p8r2_4b slurm_qwen3_cl_h200.sh es_grpo_p8_r2 4b
```

Submit its frequency-filtered counterpart with:

```bash
sbatch --job-name=q3_esgrpo_ff_p8r2_4b slurm_qwen3_cl_h200.sh es_grpo_ff_p8_r2 4b
```

## ES-GRPO anchor tuning

Filtered ES-GRPO N=16 uses a hard local ES-teacher anchor and a soft KL anchor:

- the rollout teacher is refreshed from the current student every 10 prompts;
- perturbations have `sigma=0.001` around that teacher;
- the policy loss uses `lambda * KL(pi_student || pi_base)`, with the original
  run using `lambda=0.001`.

Tune one component at a time. The coarse first pass holds refresh interval 10
and sigma 0.001 fixed while sweeping KL lambda over `0`, `0.0003`, `0.001`,
`0.003`, and `0.01`:

```bash
bash submit_anchor_sweep.sh
```

After all five runs complete, rank them using only the core validation tasks:

```bash
python select_anchor.py
```

The selector uses final core macro accuracy as the primary criterion and core
accuracy across checkpoints as a tie-breaker. It deliberately excludes AMC,
AIME, ARC-Challenge, GPQA, MedMCQA, and PubMedQA from model selection.

## Independent in-domain/out-of-domain experiment

This protocol is separate from continual learning. Each method/domain arm
starts again from Qwen3 Base and trains on only one source domain:

- math: 400 MATH-500 adaptation examples x 5 epochs = 2,000 visits;
- knowledge: 500 MMLU examples x 4 epochs = 2,000 visits;
- medical: 500 MedQA-USMLE examples x 4 epochs = 2,000 visits.

The methods are TTRL, full-parameter ES-Direct, Mode-SFT (`sft_mode` in the
submission interface), fixed-budget ECHO, EVOL-RL, and filtered ES-GRPO N=16.
The filtered ES-GRPO arm generates 16 rollouts, frequency-ranks their answers,
retains the top half (8), and applies the group-relative update to those 8.
One additional job evaluates the untouched base model. Every final checkpoint
is scored on all nine benchmarks, and every result row is marked `in_domain`
or `out_of_domain` relative to its training source.

After uploading this directory to Wolffe, submit the Qwen3-4B suite with:

```bash
cd "$HOME/experiments/EvoStrategy/qwen3_continual_baselines"
sed -i 's/\r$//' ./*.py ./*.sh
python -m py_compile common.py train.py run_continual_baselines.py \
  run_indomain_ood.py eval_indomain_ood.py show_indomain_ood_results.py
bash -n slurm_qwen3_indomain_ood_a100.sh submit_indomain_ood.sh
bash submit_indomain_ood.sh 4b
```

There are 19 jobs: one base evaluation plus six methods trained independently
on three domains. The submission helper uses both A100 partitions
(`ampere40,ampere80`), while filtered ES-GRPO is restricted to `ampere80`
because it holds three Qwen3-4B model copies during training. With four A100s,
four jobs can run concurrently and the remaining jobs wait safely in Slurm.
After jobs complete, compare all available arms with:

```bash
python show_indomain_ood_results.py
```

### Qwen3-0.6B/8B scale suite

The scale-comparison helper uses the same independent three-domain protocol
with `Qwen/Qwen3-0.6B-Base` and `Qwen/Qwen3-8B-Base`.  ES-Direct is represented
by its N=16 LoRA/vLLM implementation so all 16 perturbed adapters share one
persistent inference engine.  Mode-SFT and ES-GRPO already batch their rollout
groups; TTRL, ECHO, and EVOL-RL use prompt batches of four for 0.6B and two for
8B.  These batching changes do not change population size or prompt visits.

After both model caches are available on Wolffe, queue all 38 jobs with:

```bash
cd "$HOME/experiments/EvoStrategy/qwen3_continual_baselines"
sed -i 's/\r$//' ./*.py ./*.sh
python -m py_compile run_continual_baselines.py run_indomain_ood.py \
  eval_indomain_ood.py cache_qwen3_assets.py validate_assets.py
bash -n slurm_qwen3_indomain_ood_a100.sh submit_qwen3_scale_idood.sh
bash submit_qwen3_scale_idood.sh scale_v1
```

The 0.6B jobs can use A30, A100, H100, or H200 nodes.  The 8B jobs request
A100-80GB, H100, or H200 nodes.  Jobs submitted while those GPUs are busy stay
pending in Slurm and start automatically as resources become available.

The released ECHO comparison scripts disable DeepConf tree branching. This
suite does the same so ECHO has exactly the same 16-rollout budget as TTRL,
Mode-SFT, EVOL-RL, ES-Direct, and ES-GRPO; its entropy-confidence advantage
shaping and adaptive clipping remain enabled with the released defaults.
