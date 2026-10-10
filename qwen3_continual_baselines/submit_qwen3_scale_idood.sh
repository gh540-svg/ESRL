#!/bin/bash
# Queue the independent-domain Qwen3 scale comparison.
#
# Per backbone this submits one base evaluation plus six methods x three
# independently initialized domains.  Busy nodes are intentional: Slurm keeps
# jobs pending and starts them when a compatible GPU becomes available.
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SUITE_DIR/slurm_qwen3_indomain_ood_a100.sh"
RUN_TAG="${1:-scale_v1_$(date +%Y%m%d_%H%M%S)}"
VLLM_ENV="${VLLM_PYTHON_ENV:-$HOME/conda-envs/esdgrpo-vllm011}"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "run tag may contain only letters, digits, dot, underscore, and dash"
    exit 2
fi
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "missing Slurm script: $SLURM_SCRIPT"
    exit 1
fi
if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
    echo "missing vLLM environment: $VLLM_ENV/bin/python"
    exit 1
fi

MODELS=(0p6b 8b)
METHODS=(
    ttrl
    es_direct_lora_vllm
    sft_mode
    echo
    evol_rl
    es_grpo_filter
)
DOMAINS=(math knowledge medical)
JOBS=()

partitions_for_model() {
    case "$1" in
        0p6b) echo "ampere24,ampere40,ampere80,hopper96,hopper140" ;;
        8b)   echo "ampere80,hopper96,hopper140" ;;
        *)    return 2 ;;
    esac
}

submit_one() {
    local name="$1"
    local partitions="$2"
    shift 2
    local job
    job=$(sbatch --parsable \
        --job-name="$name" \
        --partition="$partitions" \
        --export="ALL,QWEN3_SUITE_DIR=$SUITE_DIR,VLLM_PYTHON_ENV=$VLLM_ENV" \
        "$SLURM_SCRIPT" "$@")
    JOBS+=("$job")
    printf '%-32s %-7s %-10s %s\n' "$name" "$3" "$2" "$job"
}

printf '%-32s %-7s %-10s %s\n' "job_name" "model" "domain" "job_id"
printf '%-32s %-7s %-10s %s\n' "--------------------------------" "-------" "----------" "--------"

for model in "${MODELS[@]}"; do
    partitions=$(partitions_for_model "$model")

    submit_one "q3io_base_${model}_${RUN_TAG}" "$partitions" \
        base base "$model" "$RUN_TAG" 0

    for method in "${METHODS[@]}"; do
        for domain in "${DOMAINS[@]}"; do
            short_method="$method"
            [[ "$method" == "es_direct_lora_vllm" ]] && short_method="esdl_vllm"
            [[ "$method" == "es_grpo_filter" ]] && short_method="esgrpo_ff"
            name="q3io_${short_method}_${domain}_${model}"
            submit_one "$name" "$partitions" \
                "$method" "$domain" "$model" "$RUN_TAG" 0
        done
    done
done

IFS=,
echo
echo "Submitted ${#JOBS[@]} jobs: 2 base evaluations + 36 adaptation arms."
echo "Run tag: $RUN_TAG"
echo "ES-Direct uses the persistent vLLM 0.11 LoRA engine."
echo "Other methods use their existing batched gradient-training paths."
echo "Check this submission with:"
echo "squeue -j ${JOBS[*]} -o \"%.18i %.12P %.30j %.2t %.12M %.12l %.15N %R\""
