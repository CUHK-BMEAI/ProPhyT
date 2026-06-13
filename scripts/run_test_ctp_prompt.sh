#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${ENV_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi

WORK_DIR="${WORK_DIR:-workdir}"
RUN_NAME="${RUN_NAME:-cpal_sam_dense_prompt_v2}"
DATA_PATH="${DATA_PATH:-data_penumbra_noblank_withvalid}"

MODEL_TYPE="${MODEL_TYPE:-vit_b}"
CHECKPOINT="${CHECKPOINT:-${WORK_DIR}/models/${RUN_NAME}/latest.pth}"
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
    --device "${DEVICE}"
