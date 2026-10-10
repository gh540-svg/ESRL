#!/bin/bash
set -euo pipefail

MODEL_SET="${1:-4b}"
case "$MODEL_SET" in
    4b) MODELS=(4b) ;;
    8b) MODELS=(8b) ;;
    all) MODELS=(4b 8b) ;;
    *) echo "usage: bash submit_qwen3_baselines.sh [4b|8b|all]"; exit 2 ;;
esac

METHODS=(es_direct es_direct_lora ttrl sft mode_sft evol_rl es_grpo)
declare -a JOBS=()
for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        NAME="q3_${METHOD}_${MODEL}"
        JOB_ID=$(sbatch --parsable --job-name="$NAME" \
            slurm_qwen3_cl_h200.sh "$METHOD" "$MODEL")
        JOBS+=("$JOB_ID")
        printf '%-12s %-4s %s\n' "$METHOD" "$MODEL" "$JOB_ID"
    done
done

IFS=,
echo "Check all submitted jobs with:"
echo "squeue -j ${JOBS[*]} -o \"%.18i %.12P %.24j %.2t %.12M %.15N %R\""
