#!/bin/bash
# Coarse KL-anchor sweep for filtered ES-GRPO N=16.
# Keep the hard teacher refresh interval and ES radius fixed during this pass.
set -euo pipefail

cd "$(dirname "$0")"

LAMBDAS=(0 0.0003 0.001 0.003 0.01)
TAGS=(anchor_kl0 anchor_kl0p0003 anchor_kl0p001 anchor_kl0p003 anchor_kl0p01)
JOBS=()

for index in "${!LAMBDAS[@]}"; do
    lambda="${LAMBDAS[$index]}"
    tag="${TAGS[$index]}"
    job=$(sbatch --parsable --exclusive \
        --job-name="q3_ffn16_${tag#anchor_}" \
        slurm_qwen3_cl_h200.sh \
        es_grpo_ff_n16 4b "$tag" "$lambda" 10 0.001)
    JOBS+=("$job")
    printf '%-22s lambda=%-8s job=%s\n' "$tag" "$lambda" "$job"
done

IFS=,
echo "Check all: squeue -j ${JOBS[*]} -o \"%.18i %.30j %.2t %.12M %.15N %R\""
