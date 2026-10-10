#!/bin/bash -l
#SBATCH --partition=ampere40,ampere80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

METHOD="${1:?usage: sbatch slurm_qwen3_indomain_ood_a100.sh METHOD DOMAIN MODEL_KEY [RUN_TAG [SEED]]}"
DOMAIN="${2:?usage: sbatch slurm_qwen3_indomain_ood_a100.sh METHOD DOMAIN MODEL_KEY [RUN_TAG [SEED]]}"
MODEL_KEY="${3:-4b}"
RUN_TAG="${4:-}"
SEED="${5:-0}"

case "$METHOD" in
    base|ttrl|es_direct|es_direct_lora_vllm|sft_mode|echo|evol_rl|es_grpo|es_grpo_filter|es_grpo_filter_n32) ;;
    *) echo "invalid method: $METHOD"; exit 2 ;;
esac
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid run tag: $RUN_TAG"
    exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid seed: $SEED"
    exit 2
fi
case "$DOMAIN" in
    base|math|knowledge|medical) ;;
    *) echo "invalid domain: $DOMAIN"; exit 2 ;;
esac
case "$MODEL_KEY" in
    0p6b|4b|8b) ;;
    *) echo "invalid model key: $MODEL_KEY"; exit 2 ;;
esac
if [[ "$METHOD" == "base" && "$DOMAIN" != "base" ]] || \
   [[ "$METHOD" != "base" && "$DOMAIN" == "base" ]]; then
    echo "base must pair with domain=base; trained methods need math/knowledge/medical"
    exit 2
fi

PROJECT_DIR="$HOME/experiments/EvoStrategy/ES DGRPO"
SUITE_DIR="${QWEN3_SUITE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/experiments/EvoStrategy/qwen3_continual_baselines}}"
if [[ "$METHOD" == "es_direct_lora_vllm" ]]; then
    PYTHON_ENV="${VLLM_PYTHON_ENV:-$HOME/conda-envs/esdgrpo-vllm011}"
else
    PYTHON_ENV="$HOME/conda-envs/esdgrpo310"
fi

cd "$SUITE_DIR"
module purge
module load anaconda/conda3

export PATH="$PYTHON_ENV/bin:$PATH"
export HF_HOME="$PROJECT_DIR/.hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export IDOOD_ROOT=out_qwen3_idood
export IDOOD_SNAP_EVERY=500
export IDOOD_HOURS=160
export IDOOD_MAX_NO_PROGRESS_RETRIES=3

# TTRL, ECHO, and Evol-RL can generate several prompt groups in one
# Transformers call.  Mode-SFT already batches all population rollouts, while
# ES-GRPO batches every block prompt per perturbed teacher.  These defaults
# increase utilization without changing population size or training visits.
if [[ "$MODEL_KEY" == "0p6b" ]]; then
    export QWEN3_ROLLOUT_PROMPT_BATCH_SIZE="${QWEN3_ROLLOUT_PROMPT_BATCH_SIZE:-4}"
else
    export QWEN3_ROLLOUT_PROMPT_BATCH_SIZE="${QWEN3_ROLLOUT_PROMPT_BATCH_SIZE:-2}"
fi

echo "=== Independent in-domain/out-of-domain experiment ==="
echo "job=$SLURM_JOB_ID node=$(hostname) method=$METHOD domain=$DOMAIN model=$MODEL_KEY"
echo "run_tag=${RUN_TAG:-none}; base initialization per arm; 2000 visits; seed=$SEED"
echo "rollout_prompt_batch_size=$QWEN3_ROLLOUT_PROMPT_BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ "$METHOD" == "es_direct_lora_vllm" ]]; then
    python - <<'PY'
import vllm
print("vLLM:", vllm.__version__)
if not vllm.__version__.startswith("0.11."):
    raise SystemExit("ES-Direct-LoRA-vLLM requires vLLM 0.11.x")
PY
fi

if [[ "$METHOD" == "base" ]]; then
    RESULTS="results_idood_base_qwen3_${MODEL_KEY}.json"
    if [[ -n "$RUN_TAG" ]]; then
        RESULTS="results_idood_base_qwen3_${MODEL_KEY}_${RUN_TAG}.json"
    fi
    python -u eval_indomain_ood.py \
        --base --model-key "$MODEL_KEY" --results "$RESULTS"
else
    POPULATION=16
    if [[ "$METHOD" == "es_grpo_filter_n32" ]]; then
        POPULATION=32
    fi
    ARM="${METHOD}_qwen3_${MODEL_KEY}_train_${DOMAIN}_n${POPULATION}"
    if [[ -n "$RUN_TAG" ]]; then
        ARM="${ARM}_${RUN_TAG}"
    fi
    MANIFEST="manifest_idood_${ARM}.json"
    RESULTS="results_idood_${ARM}.json"

    if [[ "$METHOD" == "es_direct" ]]; then
        FREE_GIB=$(python -c 'import shutil; print(int(shutil.disk_usage(".").free / 2**30))')
        if [[ "$FREE_GIB" -lt 150 ]]; then
            echo "ES-direct requires at least 150 GiB free for full-model snapshots"
            exit 1
        fi
    fi

    python -u run_indomain_ood.py "$METHOD" "$MODEL_KEY" "$DOMAIN" \
        --run-tag "$RUN_TAG" --seed "$SEED"
    python -u eval_indomain_ood.py \
        --manifest "$MANIFEST" --model-key "$MODEL_KEY" --results "$RESULTS"
fi

echo "[slurm] independent-domain experiment complete"
