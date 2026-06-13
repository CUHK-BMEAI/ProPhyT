#!/bin/bash
set -euo pipefail

# Subject-level 5-fold CV training wrapper for the CTP dense-prompt setup.
# Usage:
#   bash run_train_ctp_prompt_5fold.sh 0
#   bash run_train_ctp_prompt_5fold.sh all

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${ENV_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi

SOURCE_DATA="${SOURCE_DATA:-data_penumbra_noblank_withvalid}"
SPLIT_ROOT="${SPLIT_ROOT:-data_penumbra_noblank_withvalid_5fold_subject_seed42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
N_FOLDS="${N_FOLDS:-5}"

WORK_DIR="${WORK_DIR:-workdir}"
BASE_RUN_NAME="${BASE_RUN_NAME:-cpal_sam_dense_prompt_v2_cv}"

STAGE1B_CKPT="${STAGE1B_CKPT:-scripts/logs/cpal_stage1b_v2/best_model.pth}"
PROTOTYPE_BANK="${PROTOTYPE_BANK:-scripts/logs/cpal_stage1a_post/ctp_prototypes_post.npy}"
FUSION_MODE="${FUSION_MODE:-dense_prompt}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
VAL_INTERVAL="${VAL_INTERVAL:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
LATENT_DIM="${LATENT_DIM:-256}"
TOP_K="${TOP_K:-5}"
TEMPERATURE="${TEMPERATURE:-0.1}"
FREEZE_NCCT="${FREEZE_NCCT:-false}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

FOLD_SPEC="${1:-${FOLD:-all}}"

if [ ! -f "${SPLIT_ROOT}/manifest.json" ]; then
    "${PYTHON_BIN}" scripts/create_subject_5fold_splits.py \
        --data-root "${SOURCE_DATA}" \
        --output-root "${SPLIT_ROOT}" \
        --n-folds "${N_FOLDS}" \
        --seed "${SPLIT_SEED}"
fi

if [ "${FOLD_SPEC}" = "all" ]; then
    FOLDS=()
    for ((i=0; i<N_FOLDS; i++)); do
        FOLDS+=("${i}")
    done
else
    FOLDS=("${FOLD_SPEC}")
fi

for FOLD_ID in "${FOLDS[@]}"; do
    DATA_PATH="${SPLIT_ROOT}/fold${FOLD_ID}"
    RUN_NAME="${BASE_RUN_NAME}_fold${FOLD_ID}"

    if [ ! -f "${DATA_PATH}/image2label_train.json" ]; then
        echo "Missing split files for fold${FOLD_ID}: ${DATA_PATH}" >&2
        exit 1
    fi

    echo "=========================================="
    echo "CM-CPAL SAM Dense Prompt CV Train"
    echo "fold: ${FOLD_ID}"
    echo "run_name: ${RUN_NAME}"
    echo "data_path: ${DATA_PATH}"
    echo "stage1b: ${STAGE1B_CKPT}"
    echo "prototype_bank: ${PROTOTYPE_BANK}"
    echo "top_k: ${TOP_K}"
    echo "val_interval: ${VAL_INTERVAL}"
    echo "=========================================="

    CMD=(
        "${PYTHON_BIN}" train_cpal_sam_v2.py
        --work_dir "${WORK_DIR}"
        --run_name "${RUN_NAME}"
        --data_path "${DATA_PATH}"
        --stage1b_ckpt "${STAGE1B_CKPT}"
        --prototype_bank "${PROTOTYPE_BANK}"
        --fusion_mode "${FUSION_MODE}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --lr "${LR}"
        --val_interval "${VAL_INTERVAL}"
        --image_size "${IMAGE_SIZE}"
        --latent_dim "${LATENT_DIM}"
        --top_k "${TOP_K}"
        --temperature "${TEMPERATURE}"
        --device "${DEVICE}"
        --seed "${SEED}"
    )

    if [ "${FREEZE_NCCT}" = "true" ]; then
        CMD+=(--freeze_ncct)
    fi

    "${CMD[@]}"
done
