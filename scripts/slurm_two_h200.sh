#!/bin/bash -l
# One training H200 plus one remote-vLLM rollout H200 for ESRL.
# Usage: sbatch scripts/slurm_two_h200.sh DOMAIN MODEL_KEY [RUN_TAG [SEED]]

#SBATCH --partition=hopper140
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

DOMAIN="${1:?DOMAIN is required: math, knowledge, or medical}"
MODEL_KEY="${2:-4b}"
RUN_TAG="${3:-run_$(date +%Y%m%d_%H%M%S)}"
SEED="${4:-1}"

case "$MODEL_KEY" in
    0p6b) MODEL_NAME="Qwen/Qwen3-0.6B-Base" ;;
    4b) MODEL_NAME="Qwen/Qwen3-4B-Base" ;;
    8b) MODEL_NAME="Qwen/Qwen3-8B-Base" ;;
    llama3p2_3b) MODEL_NAME="meta-llama/Llama-3.2-3B" ;;
    *) echo "unsupported model key: $MODEL_KEY" >&2; exit 2 ;;
esac

case "$DOMAIN" in
    math)
        DATASET="math500"; TRAIN_SIZE=400; EVAL_SIZE=100; EPOCHS=5
        ;;
    knowledge)
        DATASET="mmlu"; TRAIN_SIZE=500; EVAL_SIZE=200; EPOCHS=4
        ;;
    medical)
        DATASET="medqa"; TRAIN_SIZE=500; EVAL_SIZE=200; EPOCHS=4
        ;;
    *) echo "unsupported domain: $DOMAIN" >&2; exit 2 ;;
esac

[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "run tag may contain only letters, numbers, dots, underscores, and dashes" >&2
    exit 2
}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be non-negative" >&2; exit 2; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_DIR/src"
PYTHON_ENV="${VLLM_PYTHON_ENV:-$HOME/conda-envs/esrl-vllm011}"
HF_CACHE="${HF_HOME:-$HOME/experiments/ESRL/.hf}"
OUTPUT_ROOT="${ESRL_OUTPUT_ROOT:-$REPO_DIR/out}"

[[ -x "$PYTHON_ENV/bin/python" ]] || {
    echo "missing Python environment: $PYTHON_ENV" >&2
    exit 1
}

module purge
module load anaconda/conda3
export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -eq 2 ]] || {
    echo "expected two H200 nodes, got: ${NODES[*]}" >&2
    exit 1
}
TRAIN_NODE="${NODES[0]}"
ROLLOUT_NODE="${NODES[1]}"

PORT=$((20000 + SLURM_JOB_ID % 20000))
AUTH="$($PYTHON_ENV/bin/python -c 'import secrets; print(secrets.token_hex(24))')"
CONTROL_DIR="$REPO_DIR/.remote_vllm"
READY_FILE="$CONTROL_DIR/${SLURM_JOB_ID}.ready.json"
SHARED_ADAPTER_DIR="$CONTROL_DIR/${SLURM_JOB_ID}.adapters"
mkdir -p "$CONTROL_DIR" "$SHARED_ADAPTER_DIR" "$OUTPUT_ROOT"
rm -f "$READY_FILE"

SERVER_STEP_PID=""
cleanup() {
    if [[ -n "$SERVER_STEP_PID" ]]; then
        kill "$SERVER_STEP_PID" 2>/dev/null || true
        wait "$SERVER_STEP_PID" 2>/dev/null || true
    fi
    rm -f "$READY_FILE"
    rmdir "$SHARED_ADAPTER_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[two-h200] training node: $TRAIN_NODE"
echo "[two-h200] rollout node:  $ROLLOUT_NODE"
echo "[config] method=esrl domain=$DOMAIN model=$MODEL_KEY seed=$SEED"
echo "[config] max_new_tokens=8196 max_model_len=12292 population=16"

(
    cd "$SRC_DIR"
    srun --nodes=1 --ntasks=1 --exclusive --nodelist="$ROLLOUT_NODE" \
        "$PYTHON_ENV/bin/python" -u vllm_remote_server.py \
        --host 0.0.0.0 --port "$PORT" --auth "$AUTH" \
        --ready-file "$READY_FILE"
) &
SERVER_STEP_PID=$!

for _ in $(seq 1 120); do
    [[ -s "$READY_FILE" ]] && break
    kill -0 "$SERVER_STEP_PID" 2>/dev/null || {
        echo "remote vLLM service exited before becoming ready" >&2
        wait "$SERVER_STEP_PID"
        exit 1
    }
    sleep 1
done
[[ -s "$READY_FILE" ]] || {
    echo "remote vLLM service did not become ready within 120 seconds" >&2
    exit 1
}

ARM="esrl_${MODEL_KEY}_train_${DOMAIN}_n16_${RUN_TAG}_seed${SEED}"
OUTPUT="$OUTPUT_ROOT/$ARM"
ARGS=(
    --model "$MODEL_NAME"
    --dataset "$DATASET"
    --out "$OUTPUT"
    --train-size "$TRAIN_SIZE"
    --eval-size "$EVAL_SIZE"
    --epochs "$EPOCHS"
    --population 16
    --rollouts-per-member 1
    --temp 0.8
    --max-new 8196
    --p-high 0.95
    --lora-r 16
    --lr 1e-5
    --kl-lambda 0.001
    --block-size 10
    --prompt-update-batch-size 2
    --sigma 0.001
    --snap-every 500
    --max-visits 2000
    --dtype bfloat16
    --seed "$SEED"
    --data-seed 0
    --rollout-prompt-batch-size 2
    --max-model-len 12292
    --max-num-seqs 32
    --gpu-memory-utilization 0.40
)
if [[ "$DOMAIN" != math ]]; then
    ARGS+=(--permute-choices)
fi

cd "$SRC_DIR"
srun --nodes=1 --ntasks=1 --exclusive --nodelist="$TRAIN_NODE" \
    env \
    PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_REMOTE_HOST="$ROLLOUT_NODE" \
    VLLM_REMOTE_PORT="$PORT" \
    VLLM_REMOTE_AUTH="$AUTH" \
    VLLM_REMOTE_CONNECT_TIMEOUT=300 \
    VLLM_SHARED_TMPDIR="$SHARED_ADAPTER_DIR" \
    "$PYTHON_ENV/bin/python" -u train_esrl.py "${ARGS[@]}"

echo "[two-h200] complete: $ARM"
