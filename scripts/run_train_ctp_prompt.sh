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

STAGE1B_CKPT="${STAGE1B_CKPT:-scripts/logs/cpal_stage1b_v2/best_model.pth}"
PROTOTYPE_BANK="${PROTOTYPE_BANK:-scripts/logs/cpal_stage1a_post/ctp_prototypes_post.npy}"
FUSION_MODE="${FUSION_MODE:-dense_prompt}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
LATENT_DIM="${LATENT_DIM:-256}"
TOP_K="${TOP_K:-5}"
TEMPERATURE="${TEMPERATURE:-0.1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
FREEZE_NCCT="${FREEZE_NCCT:-false}"

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
