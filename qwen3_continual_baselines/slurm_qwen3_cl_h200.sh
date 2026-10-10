#!/bin/bash -l
#SBATCH --partition=hopper140
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

METHOD="${1:?usage: sbatch slurm_qwen3_cl_h200.sh METHOD MODEL_KEY [RUN_TAG [KL_LAMBDA [ANCHOR_BLOCK_SIZE [ES_SIGMA]]]]}"
MODEL_KEY="${2:?usage: sbatch slurm_qwen3_cl_h200.sh METHOD MODEL_KEY [RUN_TAG [KL_LAMBDA [ANCHOR_BLOCK_SIZE [ES_SIGMA]]]]}"
RUN_TAG="${3:-}"
KL_LAMBDA="${4:-0.001}"
ANCHOR_BLOCK_SIZE="${5:-10}"
ES_SIGMA="${6:-0.001}"
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid run tag: $RUN_TAG"
    exit 2
fi
case "$METHOD" in
    es_direct|es_direct_lora|es_direct_lora_batch16|es_direct_lora_vllm|ttrl|sft|mode_sft|echo|evol_rl|es_grpo|es_grpo_p8_r2|es_grpo_ff_p8_r2|es_grpo_ff_n16|es_grpo_ff_n32|es_grpo_ff_n16_tok1024) ;;
    *) echo "invalid method: $METHOD"; exit 2 ;;
esac
case "$MODEL_KEY" in
    4b|8b) ;;
    *) echo "invalid model key: $MODEL_KEY"; exit 2 ;;
esac

PROJECT_DIR="$HOME/experiments/EvoStrategy/ES DGRPO"
# Slurm copies the submitted script into a spool directory before executing it,
# so BASH_SOURCE[0] does NOT identify the project directory inside a batch job.
# SLURM_SUBMIT_DIR is the directory from which `sbatch` was invoked; both the
# direct commands and the submission helpers intentionally submit from there.
SUITE_DIR="${QWEN3_SUITE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/experiments/EvoStrategy/qwen3_continual_baselines}}"
PYTHON_ENV="${VLLM_PYTHON_ENV:-$HOME/conda-envs/esdgrpo310}"

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
export PYTHONPATH="$SUITE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CL_SNAP_EVERY=500
export CL_HOURS_PER_STAGE=55
export CL_MAX_NO_PROGRESS_RETRIES=3

# vLLM 0.11 exposes LLMEngine through V1. The worker extension is injected
# into its V1 worker and reached through collective_rpc.
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

case "$METHOD" in
    sft)
        ARM_SUFFIX="sft_qwen3_${MODEL_KEY}"
        ;;
    es_grpo_ff_n16)
        ARM_SUFFIX="es_grpo_ff_qwen3_${MODEL_KEY}_n16"
        ;;
    es_grpo_ff_n32)
        ARM_SUFFIX="es_grpo_ff_qwen3_${MODEL_KEY}_n32"
        ;;
    es_grpo_ff_n16_tok1024)
        ARM_SUFFIX="es_grpo_ff_tok1024_qwen3_${MODEL_KEY}_n16"
        ;;
    es_grpo_p8_r2)
        ARM_SUFFIX="es_grpo_p8_r2_qwen3_${MODEL_KEY}"
        ;;
    es_grpo_ff_p8_r2)
        ARM_SUFFIX="es_grpo_ff_p8_r2_qwen3_${MODEL_KEY}"
        ;;
    *)
        ARM_SUFFIX="${METHOD}_qwen3_${MODEL_KEY}_n16"
        ;;
esac
if [ -n "$RUN_TAG" ]; then
    ARM_SUFFIX="${ARM_SUFFIX}_${RUN_TAG}"
fi
MANIFEST="manifest_${ARM_SUFFIX}.json"
RESULTS="results_${ARM_SUFFIX}.json"

echo "=== Qwen3 continual baseline ==="
echo "job=$SLURM_JOB_ID node=$(hostname) method=$METHOD model=$MODEL_KEY"
echo "order=MATH-500 -> MMLU -> MedQA-USMLE; 2000 visits/stage"
echo "run_tag=${RUN_TAG:-none} kl_lambda=$KL_LAMBDA anchor_block_size=$ANCHOR_BLOCK_SIZE es_sigma=$ES_SIGMA"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import shutil
import torch
import transformers
import peft
from pathlib import Path

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
free = shutil.disk_usage(Path.cwd()).free / 2**30
print(f"Free shared storage: {free:.1f} GiB")
PY

if [ "$METHOD" = "es_direct_lora_vllm" ]; then
    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    found = version("vllm")
except PackageNotFoundError as error:
    raise SystemExit(
        "vLLM is missing. Install requirements_vllm_es.txt in VLLM_PYTHON_ENV."
    ) from error
if not found.startswith("0.11."):
    raise SystemExit(f"expected vLLM 0.11.x, found {found}")
print("vLLM:", found)
PY
fi

if [ "$METHOD" = "es_direct" ]; then
    FREE_GIB=$(python -c 'import shutil; print(int(shutil.disk_usage(".").free / 2**30))')
    if [ "$FREE_GIB" -lt 300 ]; then
        echo "ES-direct requires at least 300 GiB free for full-model checkpoints"
        exit 1
    fi
fi

python -u run_continual_baselines.py "$METHOD" "$MODEL_KEY" \
    --run-tag "$RUN_TAG" \
    --kl-lambda "$KL_LAMBDA" \
    --anchor-block-size "$ANCHOR_BLOCK_SIZE" \
    --es-sigma "$ES_SIGMA"
python -u eval_continual_qwen3.py \
    --manifest "$MANIFEST" \
    --results "$RESULTS"

echo "[slurm] complete: $ARM_SUFFIX"
