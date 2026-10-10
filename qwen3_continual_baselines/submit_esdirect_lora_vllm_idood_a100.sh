#!/bin/bash
# Run one independent ES-Direct-LoRA-vLLM arm on each currently free A100.
set -euo pipefail

MODEL_KEY="${1:-4b}"
RUN_TAG="${2:-esdlvllm_a100_$(date +%Y%m%d_%H%M%S)}"
SLURM_SCRIPT="slurm_qwen3_indomain_ood_a100.sh"
VLLM_ENV="${VLLM_PYTHON_ENV:-$HOME/conda-envs/esdgrpo-vllm011}"

case "$MODEL_KEY" in
    4b|8b) ;;
    *) echo "usage: bash submit_esdirect_lora_vllm_idood_a100.sh [4b|8b] [RUN_TAG]"; exit 2 ;;
esac
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid run tag: $RUN_TAG"
    exit 2
fi
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Run this script from qwen3_continual_baselines; missing $SLURM_SCRIPT"
    exit 2
fi
if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
    echo "vLLM Python environment is missing: $VLLM_ENV/bin/python"
    exit 2
fi

DOMAINS=(math knowledge medical)
PARTITIONS=(ampere80 ampere80 ampere40)
NODES=(a100-100 a100-101 a100-002)
JOBS=()

for index in "${!DOMAINS[@]}"; do
    domain="${DOMAINS[$index]}"
    partition="${PARTITIONS[$index]}"
    node="${NODES[$index]}"
    state=$(sinfo -N -h -n "$node" -o "%T" | head -n 1)
    echo "$domain target: $partition/$node (current state: ${state:-unknown})"
    name="q3io_esdlv_${domain}_${MODEL_KEY}"
    job=$(sbatch --parsable \
        --partition="$partition" \
        --nodelist="$node" \
        --job-name="$name" \
        --export="ALL,VLLM_PYTHON_ENV=$VLLM_ENV" \
        "$SLURM_SCRIPT" es_direct_lora_vllm "$domain" \
        "$MODEL_KEY" "$RUN_TAG" 0)
    JOBS+=("$job")
    printf '%-10s %-9s %-10s job=%s\n' "$domain" "$partition" "$node" "$job"
done

joined=$(IFS=,; echo "${JOBS[*]}")
echo
echo "Submitted three independent ES-Direct-LoRA-vLLM jobs; run_tag=$RUN_TAG"
squeue -j "$joined" -o "%.18i %.12P %.30j %.2t %.12M %.15N %R"
