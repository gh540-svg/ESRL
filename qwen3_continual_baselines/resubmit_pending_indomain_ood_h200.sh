#!/bin/bash
# Move only the known pending independent-domain jobs from A100 queues to H200.
# Running jobs are deliberately left untouched.
set -euo pipefail

MODEL_KEY="4b"
SLURM_SCRIPT="slurm_qwen3_indomain_ood_a100.sh"
ROLLOUT_PROMPT_BATCH_SIZE="${QWEN3_ROLLOUT_PROMPT_BATCH_SIZE:-4}"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Run this script from qwen3_continual_baselines; missing $SLURM_SCRIPT"
    exit 2
fi

OLD_JOBS=(108737 108738 108739 108740 108741 108742 108743 108744 108745 108746 108747)
METHODS=(sft_mode sft_mode echo echo echo evol_rl evol_rl evol_rl es_grpo_filter es_grpo_filter es_grpo_filter)
DOMAINS=(knowledge medical math knowledge medical math knowledge medical math knowledge medical)

NEW_JOBS=()
for index in "${!OLD_JOBS[@]}"; do
    old_job="${OLD_JOBS[$index]}"
    method="${METHODS[$index]}"
    domain="${DOMAINS[$index]}"
    state=$(squeue -h -j "$old_job" -o "%T" 2>/dev/null || true)
    if [[ "$state" != "PENDING" ]]; then
        echo "skip old job $old_job ($method/$domain): state=${state:-not active}"
        continue
    fi

    # Hold first so a job cannot start on an A100 between the state check and
    # cancellation. If it already left the queue, leave it alone.
    if ! scontrol hold "$old_job" 2>/dev/null; then
        echo "skip old job $old_job ($method/$domain): could not hold it"
        continue
    fi
    state=$(squeue -h -j "$old_job" -o "%T" 2>/dev/null || true)
    if [[ "$state" != "PENDING" ]]; then
        scontrol release "$old_job" 2>/dev/null || true
        echo "skip old job $old_job ($method/$domain): became ${state:-inactive}"
        continue
    fi

    name="q3io_${method}_${domain}_${MODEL_KEY}"
    if ! new_job=$(sbatch --parsable --hold \
        --partition=hopper140 \
        --job-name="$name" \
        --export="ALL,QWEN3_ROLLOUT_PROMPT_BATCH_SIZE=$ROLLOUT_PROMPT_BATCH_SIZE" \
        "$SLURM_SCRIPT" "$method" "$domain" "$MODEL_KEY"); then
        scontrol release "$old_job" 2>/dev/null || true
        echo "replacement submission failed; released old job $old_job"
        continue
    fi
    scancel "$old_job"
    scontrol release "$new_job"
    NEW_JOBS+=("$new_job")
    printf 'moved %-8s %-10s old=%s new=%s\n' "$method" "$domain" "$old_job" "$new_job"
done

if ((${#NEW_JOBS[@]})); then
    joined=$(IFS=,; echo "${NEW_JOBS[*]}")
    echo
    echo "Submitted ${#NEW_JOBS[@]} H200 jobs; rollout prompt batch=$ROLLOUT_PROMPT_BATCH_SIZE"
    squeue -j "$joined" -o "%.18i %.12P %.30j %.2t %.12M %.15N %R"
else
    echo "No listed jobs were still pending; nothing was cancelled or submitted."
fi
