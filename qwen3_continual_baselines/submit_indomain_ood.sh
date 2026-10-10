#!/bin/bash
set -euo pipefail

MODEL_KEY="${1:-4b}"
case "$MODEL_KEY" in
    4b|8b) ;;
    *) echo "usage: bash submit_indomain_ood.sh [4b|8b]"; exit 2 ;;
esac

METHODS=(ttrl es_direct sft_mode echo evol_rl es_grpo_filter)
DOMAINS=(math knowledge medical)
JOBS=()

BASE_JOB=$(sbatch --parsable --job-name="q3io_base_${MODEL_KEY}" \
    --partition=ampere40,ampere80 \
    slurm_qwen3_indomain_ood_a100.sh base base "$MODEL_KEY")
JOBS+=("$BASE_JOB")
printf '%-12s %-10s %-4s %s\n' base base "$MODEL_KEY" "$BASE_JOB"

for METHOD in "${METHODS[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        NAME="q3io_${METHOD}_${DOMAIN}_${MODEL_KEY}"
        if [[ "$METHOD" == "es_grpo_filter" ]]; then
            PARTITIONS="ampere80"
        else
            PARTITIONS="ampere40,ampere80"
        fi
        JOB_ID=$(sbatch --parsable --job-name="$NAME" \
            --partition="$PARTITIONS" \
            slurm_qwen3_indomain_ood_a100.sh "$METHOD" "$DOMAIN" "$MODEL_KEY")
        JOBS+=("$JOB_ID")
        printf '%-12s %-10s %-4s %s\n' "$METHOD" "$DOMAIN" "$MODEL_KEY" "$JOB_ID"
    done
done

IFS=,
echo
echo "Submitted ${#JOBS[@]} jobs (1 base + 18 adaptation arms)."
echo "Check them with:"
echo "squeue -j ${JOBS[*]} -o \"%.18i %.12P %.30j %.2t %.12M %.15N %R\""
