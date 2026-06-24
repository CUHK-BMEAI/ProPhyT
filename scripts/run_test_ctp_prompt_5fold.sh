#!/bin/bash
set -euo pipefail

# Subject-level 5-fold CV testing wrapper for the CTP dense-prompt setup.
# Usage:
#   bash scripts/run_test_ctp_prompt_5fold.sh 0
#   bash scripts/run_test_ctp_prompt_5fold.sh all

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

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
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/cv_results}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-latest.pth}"

MODEL_TYPE="${MODEL_TYPE:-vit_b}"
STAGE1B_CKPT="${STAGE1B_CKPT:-scripts/logs/cpal_stage1b_v2/best_model.pth}"
PROTOTYPE_BANK="${PROTOTYPE_BANK:-scripts/logs/cpal_stage1a_post/ctp_prototypes_post.npy}"
FUSION_MODE="${FUSION_MODE:-dense_prompt}"
LATENT_DIM="${LATENT_DIM:-256}"
TOP_K="${TOP_K:-5}"
TEMPERATURE="${TEMPERATURE:-0.1}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BOXES_PROMPT="${BOXES_PROMPT:-True}"
POINT_NUM="${POINT_NUM:-1}"
ITER_POINT="${ITER_POINT:-1}"
SAVE_PRED="${SAVE_PRED:-True}"
DEVICE="${DEVICE:-cuda}"

FOLD_SPEC="${1:-${FOLD:-all}}"

if [ ! -f "${SPLIT_ROOT}/manifest.json" ]; then
    "${PYTHON_BIN}" scripts/create_subject_5fold_splits.py \
        --data-root "${SOURCE_DATA}" \
        --output-root "${SPLIT_ROOT}" \
        --n-folds "${N_FOLDS}" \
        --seed "${SPLIT_SEED}"
fi

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
    DATA_PATH="${SPLIT_ROOT}/fold${FOLD_ID}"
    RUN_NAME="${BASE_RUN_NAME}_fold${FOLD_ID}"
    CHECKPOINT="${WORK_DIR}/models/${RUN_NAME}/${CHECKPOINT_NAME}"
    OUTPUT_CSV="${RESULT_DIR}/${RUN_NAME}_results_cpal_sam.csv"

    if [ ! -f "${DATA_PATH}/label2image_test.json" ]; then
        echo "Missing split files for fold${FOLD_ID}: ${DATA_PATH}" >&2
        exit 1
    fi
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "Missing checkpoint for fold${FOLD_ID}: ${CHECKPOINT}" >&2
        exit 1
    fi

    echo "=========================================="
    echo "CM-CPAL SAM Dense Prompt CV Test"
    echo "fold: ${FOLD_ID}"
    echo "run_name: ${RUN_NAME}"
    echo "data_path: ${DATA_PATH}"
    echo "checkpoint: ${CHECKPOINT}"
    echo "output_csv: ${OUTPUT_CSV}"
    echo "=========================================="

    "${PYTHON_BIN}" test_cpal_sam_v2.py \
        --work_dir "${WORK_DIR}" \
        --run_name "${RUN_NAME}" \
        --data_path "${DATA_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --checkpoint "${CHECKPOINT}" \
        --stage1b_ckpt "${STAGE1B_CKPT}" \
        --prototype_bank "${PROTOTYPE_BANK}" \
        --fusion_mode "${FUSION_MODE}" \
        --latent_dim "${LATENT_DIM}" \
        --top_k "${TOP_K}" \
        --temperature "${TEMPERATURE}" \
        --image_size "${IMAGE_SIZE}" \
        --boxes_prompt "${BOXES_PROMPT}" \
        --point_num "${POINT_NUM}" \
        --iter_point "${ITER_POINT}" \
        --save_pred "${SAVE_PRED}" \
        --output_csv "${OUTPUT_CSV}" \
        --device "${DEVICE}"
done

if [ "${FOLD_SPEC}" = "all" ]; then
    "${PYTHON_BIN}" scripts/summarize_cpal_cv_results.py \
        --result-dir "${RESULT_DIR}" \
        --base-run-name "${BASE_RUN_NAME}" \
        --n-folds "${N_FOLDS}"
fi
