#!/bin/bash
# Submit three ES-GRPO configurations across three independent training domains.
set -euo pipefail

MODEL_KEY="${1:-4b}"
RUN_TAG="${2:-esgrid_$(date +%Y%m%d_%H%M%S)}"
SLURM_SCRIPT="slurm_qwen3_indomain_ood_a100.sh"

case "$MODEL_KEY" in
    4b|8b) ;;
    *) echo "usage: bash submit_esgrpo_idood_matrix_h200.sh [4b|8b] [RUN_TAG]"; exit 2 ;;
esac
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid run tag: $RUN_TAG"
    exit 2
fi
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Run this script from qwen3_continual_baselines; missing $SLURM_SCRIPT"
    exit 2
fi

DOMAINS=(math knowledge medical)
JOBS=()

# Three genuinely independent filtered-N16 reruns. Seeds 1,2,3 avoid
# duplicating the already-completed seed-0 result.
for seed in 1 2 3; do
    for domain in "${DOMAINS[@]}"; do
        repeat_tag="${RUN_TAG}_rep${seed}"
        name="q3io_ff16r${seed}_${domain}_${MODEL_KEY}"
        job=$(sbatch --parsable \
            --partition=hopper140 \
            --job-name="$name" \
            "$SLURM_SCRIPT" es_grpo_filter "$domain" "$MODEL_KEY" \
            "$repeat_tag" "$seed")
        JOBS+=("$job")
        printf '%-22s seed=%-2s %-10s job=%s\n' \
            es_grpo_filter "$seed" "$domain" "$job"
    done
done

# One seed-0 comparison for unfiltered N16 and filtered N32 on each domain.
for method in es_grpo es_grpo_filter_n32; do
    case "$method" in
        es_grpo) short="raw16" ;;
        es_grpo_filter_n32) short="ff32" ;;
    esac
    for domain in "${DOMAINS[@]}"; do
        name="q3io_${short}_${domain}_${MODEL_KEY}"
        job=$(sbatch --parsable \
            --partition=hopper140 \
            --job-name="$name" \
            "$SLURM_SCRIPT" "$method" "$domain" "$MODEL_KEY" \
            "$RUN_TAG" 0)
        JOBS+=("$job")
        printf '%-22s seed=0  %-10s job=%s\n' "$method" "$domain" "$job"
    done
done

joined=$(IFS=,; echo "${JOBS[*]}")
echo
echo "Submitted ${#JOBS[@]} independent H200 jobs; run_tag=$RUN_TAG"
squeue -j "$joined" -o "%.18i %.12P %.30j %.2t %.12M %.15N %R"
