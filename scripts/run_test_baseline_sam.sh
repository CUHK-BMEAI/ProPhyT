#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${ENV_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi

WORK_DIR="${WORK_DIR:-workdir}"
RUN_NAME="${RUN_NAME:-baseline_sam}"
DATA_PATH="${DATA_PATH:-data_penumbra_noblank_withvalid}"

MODEL_TYPE="${MODEL_TYPE:-vit_b}"
CHECKPOINT="${CHECKPOINT:-${WORK_DIR}/models/${RUN_NAME}/best_dice_sam.pth}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
BOXES_PROMPT="${BOXES_PROMPT:-True}"
POINT_NUM="${POINT_NUM:-1}"
ITER_POINT="${ITER_POINT:-8}"
SAVE_PRED="${SAVE_PRED:-True}"
DEVICE="${DEVICE:-cuda}"

"${PYTHON_BIN}" test.py \
    --work_dir "${WORK_DIR}" \
    --run_name "${RUN_NAME}" \
    --data_path "${DATA_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --sam_checkpoint "${CHECKPOINT}" \
    --image_size "${IMAGE_SIZE}" \
    --boxes_prompt "${BOXES_PROMPT}" \
    --point_num "${POINT_NUM}" \
    --iter_point "${ITER_POINT}" \
    --save_pred "${SAVE_PRED}" \
    --device "${DEVICE}"
