#!/bin/bash -l
# Verify/install the uploaded Qwen3 scale cache, validate the offline suite, and
# optionally submit the complete independent-domain experiment matrix.
set -euo pipefail

RUN_TAG="${1:-scale_v1}"
ACTION="${2:---validate-only}"

case "$ACTION" in
    --validate-only|--submit) ;;
    *) echo "usage: $0 [RUN_TAG] [--validate-only|--submit]"; exit 2 ;;
esac
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "run tag may contain only letters, digits, dot, underscore, and dash"
    exit 2
fi

USER_ROOT="/bigdata/users/22173342"
ARCHIVE="$USER_ROOT/qwen3_scale_models.tar"
CHECKSUM="$USER_ROOT/qwen3_scale_models.tar.sha256"
SUITE_DIR="$HOME/experiments/EvoStrategy/qwen3_continual_baselines"
PROJECT_DIR="$HOME/experiments/EvoStrategy/ES DGRPO"
PYTHON_ENV="$HOME/conda-envs/esdgrpo310"
VLLM_ENV="$HOME/conda-envs/esdgrpo-vllm011"
export HF_HOME="$PROJECT_DIR/.hf"

required_files=(
    "$ARCHIVE"
    "$CHECKSUM"
    "$SUITE_DIR/run_continual_baselines.py"
    "$SUITE_DIR/run_indomain_ood.py"
    "$SUITE_DIR/eval_indomain_ood.py"
    "$SUITE_DIR/validate_assets.py"
    "$SUITE_DIR/slurm_qwen3_indomain_ood_a100.sh"
    "$SUITE_DIR/submit_qwen3_scale_idood.sh"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "missing required file: $path"
        exit 1
    fi
done

echo "=== 1/6 Verify uploaded archive ==="
cd "$USER_ROOT"
sha256sum -c "$(basename "$CHECKSUM")"
tar -tf "$ARCHIVE" >/dev/null

echo "=== 2/6 Extract model caches ==="
mkdir -p "$HF_HOME"
tar -xf "$ARCHIVE" -C "$HF_HOME"
du -sh \
    "$HF_HOME/hub/models--Qwen--Qwen3-0.6B-Base" \
    "$HF_HOME/hub/models--Qwen--Qwen3-8B-Base"

echo "=== 3/6 Validate uploaded experiment code ==="
cd "$SUITE_DIR"
sed -i 's/\r$//' ./*.py ./*.sh

module purge
module load anaconda/conda3
export PATH="$PYTHON_ENV/bin:$PATH"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python -m py_compile \
    run_continual_baselines.py \
    run_indomain_ood.py \
    eval_indomain_ood.py \
    validate_assets.py
bash -n slurm_qwen3_indomain_ood_a100.sh submit_qwen3_scale_idood.sh

echo "=== 4/6 Validate both offline model/dataset configurations ==="
python -u validate_assets.py 0p6b
python -u validate_assets.py 8b

echo "=== 5/6 Validate vLLM environment ==="
if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
    echo "missing vLLM Python: $VLLM_ENV/bin/python"
    exit 1
fi
"$VLLM_ENV/bin/python" - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
if not vllm.__version__.startswith("0.11."):
    raise SystemExit("expected vLLM 0.11.x")
PY

echo "=== 6/6 Validation complete ==="
if [[ "$ACTION" == "--validate-only" ]]; then
    echo "No jobs submitted. To submit after reviewing validation, run:"
    echo "  $0 $RUN_TAG --submit"
    exit 0
fi

RECEIPT="$SUITE_DIR/submitted_qwen3_scale_${RUN_TAG}.txt"
if [[ -e "$RECEIPT" ]]; then
    echo "submission receipt already exists: $RECEIPT"
    echo "Refusing to submit duplicate jobs. Use a new run tag if intentional."
    exit 1
fi

export VLLM_PYTHON_ENV="$VLLM_ENV"
bash submit_qwen3_scale_idood.sh "$RUN_TAG" | tee "$RECEIPT"

echo
echo "Submission receipt: $RECEIPT"
squeue -u "$USER" \
    -o "%.18i %.12P %.30j %.2t %.12M %.12l %.15N %R"
