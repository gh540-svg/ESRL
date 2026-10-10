#!/bin/bash
# Submit the three-method, four-model, three-domain matrix.
# ESRL uses seeds 1/2/3; ES-Direct and SFT use seed 0.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_SCRIPT="$REPO_DIR/scripts/slurm_two_h200.sh"
RUN_TAG="${1:-tok8196_$(date +%Y%m%d_%H%M%S)}"
RECEIPT="$REPO_DIR/submitted_${RUN_TAG}.tsv"

MODELS=(0p6b 4b 8b llama3p2_3b)
DOMAINS=(math knowledge medical)

printf 'method\tmodel\tdomain\tseed\tmax_new_tokens\tmax_model_len\tgpus\tjobid\n' > "$RECEIPT"

submit_one() {
    local method="$1" model="$2" domain="$3" seed="$4"
    local name="${method}_${model}_${domain}_s${seed}"
    local job
    job="$(sbatch --parsable --job-name="$name" \
        "$SLURM_SCRIPT" "$method" "$domain" "$model" "$RUN_TAG" "$seed")"
    job="${job%%;*}"
    printf '%s\t%s\t%s\t%s\t8196\t12292\t2xH200\t%s\n' \
        "$method" "$model" "$domain" "$seed" "$job" | tee -a "$RECEIPT"
}

for model in "${MODELS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        submit_one es_direct "$model" "$domain" 0
        submit_one sft "$model" "$domain" 0
        for seed in 1 2 3; do
            submit_one esrl "$model" "$domain" "$seed"
        done
    done
done

echo
echo "Submitted 60 jobs. Receipt: $RECEIPT"
