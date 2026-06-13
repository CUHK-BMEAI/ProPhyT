#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${ENV_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi

WORK_DIR="${WORK_DIR:-workdir}"
RUN_NAME="${RUN_NAME:-baseline_sam}"
DATA_PATH="${DATA_PATH:-data_penumbra_noblank_withvalid}"

MODEL_TYPE="${MODEL_TYPE:-vit_b}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-sam-med2d_b1106.pth}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"

"${PYTHON_BIN}" train.py \
    --work_dir "${WORK_DIR}" \
    --run_name "${RUN_NAME}" \
    --data_path "${DATA_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --sam_checkpoint "${SAM_CHECKPOINT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --image_size "${IMAGE_SIZE}" \
    --device "${DEVICE}"
