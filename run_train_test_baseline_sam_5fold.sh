#!/bin/bash
set -euo pipefail

# Subject-level 5-fold CV wrapper for the plain SAM-Med2D fine-tuning baseline.
# Usage:
#   GPU_ID=0 bash run_train_test_baseline_sam_5fold.sh all
#   GPU_ID=0 bash run_train_test_baseline_sam_5fold.sh 4

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-swin_litemedsam}"
CONDA_SH="${CONDA_SH:-}"
if [ -n "${CONDA_SH}" ] && [ -f "${CONDA_SH}" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
if [ -n "${ENV_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi

SPLIT_COMPAT="${SPLIT_COMPAT:-${REPO_ROOT}/data_penumbra_noblank_withvalid_5fold_subject_seed42}"
N_FOLDS="${N_FOLDS:-5}"

WORK_DIR="${WORK_DIR:-workdir}"
BASE_RUN_NAME="${BASE_RUN_NAME:-baseline_sam_cv}"
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/cv_results}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_dice_sam.pth}"

MODEL_TYPE="${MODEL_TYPE:-vit_b}"
INIT_SAM_CHECKPOINT="${INIT_SAM_CHECKPOINT:-sam-med2d_b1106.pth}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"

BOXES_PROMPT="${BOXES_PROMPT:-True}"
POINT_NUM="${POINT_NUM:-1}"
ITER_POINT_TEST="${ITER_POINT_TEST:-1}"
SAVE_PRED="${SAVE_PRED:-True}"

FOLD_SPEC="${1:-${FOLD:-all}}"

"${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch

print(f"python: {sys.executable}")
print(f"torch: {torch.__version__}, torch_cuda: {torch.version.cuda}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
try:
    cuda_ok = torch.cuda.is_available()
except Exception as exc:
    print(f"CUDA check failed: {exc}", file=sys.stderr)
    sys.exit(10)
if not cuda_ok:
    print("CUDA check failed: torch.cuda.is_available() is False", file=sys.stderr)
    sys.exit(10)
print(f"cuda_device0: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "${RESULT_DIR}"

if [ "${FOLD_SPEC}" = "all" ]; then
    FOLDS=()
    for ((i=0; i<N_FOLDS; i++)); do
        FOLDS+=("${i}")
    done
else
    FOLDS=("${FOLD_SPEC}")
fi

for FOLD_ID in "${FOLDS[@]}"; do
    DATA_PATH="${SPLIT_COMPAT}/fold${FOLD_ID}"
    RUN_NAME="${BASE_RUN_NAME}_fold${FOLD_ID}"
    CHECKPOINT="${WORK_DIR}/models/${RUN_NAME}/${CHECKPOINT_NAME}"
    OUTPUT_CSV="${RESULT_DIR}/${RUN_NAME}_results.csv"

    if [ ! -f "${DATA_PATH}/image2label_train.json" ] || [ ! -f "${DATA_PATH}/label2image_test.json" ]; then
        echo "Missing split files for fold${FOLD_ID}: ${DATA_PATH}" >&2
        exit 1
    fi

    echo "=========================================="
    echo "SAM-Med2D FT-P CV"
    echo "fold: ${FOLD_ID}"
    echo "conda_env: ${CONDA_ENV}"
    echo "gpu_id: ${GPU_ID:-0}"
    echo "run_name: ${RUN_NAME}"
    echo "data_path: ${DATA_PATH}"
    echo "checkpoint: ${CHECKPOINT}"
    echo "output_csv: ${OUTPUT_CSV}"
    echo "=========================================="

    if [ ! -f "${CHECKPOINT}" ]; then
        "${PYTHON_BIN}" train.py \
            --work_dir "${WORK_DIR}" \
            --run_name "${RUN_NAME}" \
            --data_path "${DATA_PATH}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --image_size "${IMAGE_SIZE}" \
            --device "${DEVICE}" \
            --model_type "${MODEL_TYPE}" \
            --sam_checkpoint "${INIT_SAM_CHECKPOINT}"
    else
        echo "Found existing checkpoint, skip training: ${CHECKPOINT}"
    fi

    if [ ! -f "${CHECKPOINT}" ]; then
        echo "Training finished but checkpoint is missing: ${CHECKPOINT}" >&2
        exit 1
    fi

    "${PYTHON_BIN}" test.py \
        --work_dir "${WORK_DIR}" \
        --run_name "${RUN_NAME}" \
        --data_path "${DATA_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --sam_checkpoint "${CHECKPOINT}" \
        --image_size "${IMAGE_SIZE}" \
        --boxes_prompt "${BOXES_PROMPT}" \
        --point_num "${POINT_NUM}" \
        --iter_point "${ITER_POINT_TEST}" \
        --save_pred "${SAVE_PRED}" \
        --output_csv "${OUTPUT_CSV}" \
        --device "${DEVICE}"
done
