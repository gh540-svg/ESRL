#!/bin/bash
# Submit ESRL seed 1 for four models and three independent training domains.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_SCRIPT="$REPO_DIR/scripts/slurm_two_h200.sh"
RUN_TAG="${1:-tok8196_$(date +%Y%m%d_%H%M%S)}"
SEED=1
RECEIPT="$REPO_DIR/submitted_${RUN_TAG}.tsv"

MODELS=(0p6b 4b 8b llama3p2_3b)
DOMAINS=(math knowledge medical)

printf 'method\tmodel\tdomain\tseed\tmax_new_tokens\tmax_model_len\tgpus\tjobid\n' > "$RECEIPT"

for model in "${MODELS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        name="esrl_${model}_${domain}_s${SEED}"
        job="$(sbatch --parsable --job-name="$name" \
            "$SLURM_SCRIPT" "$domain" "$model" "$RUN_TAG" "$SEED")"
        job="${job%%;*}"
        printf 'esrl\t%s\t%s\t%s\t8196\t12292\t2xH200\t%s\n' \
            "$model" "$domain" "$SEED" "$job" | tee -a "$RECEIPT"
    done
done

echo
echo "Submitted 12 ESRL seed-1 jobs. Receipt: $RECEIPT"
