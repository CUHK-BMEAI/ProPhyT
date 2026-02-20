#!/bin/bash



export LD_LIBRARY_PATH=/home/danying/miniconda3/envs/brain/lib:$LD_LIBRARY_PATH


WORK_DIR="workdir"
RUN_NAME="cpal_sam_dense_prompt_v2"
DATA_PATH="data_penumbra_noblank_withvalid"

STAGE1B_CKPT="scripts/logs/cpal_stage1b_v2/best_model.pth"
PROTOTYPE_BANK="scripts/logs/cpal_stage1a_post/ctp_prototypes_post.npy"


FUSION_MODE="dense_prompt"


EPOCHS=15
BATCH_SIZE=2
LR=1e-4
IMAGE_SIZE=256

#
LATENT_DIM=256
TOP_K=5
TEMPERATURE=0.1


FREEZE_NCCT=false  #微调fpn，就是哪个ncctencoder


CMD="python train_cpal_sam_v2.py \
    --work_dir ${WORK_DIR} \
    --run_name ${RUN_NAME} \
    --data_path ${DATA_PATH} \
    --stage1b_ckpt ${STAGE1B_CKPT} \
    --prototype_bank ${PROTOTYPE_BANK} \
    --fusion_mode ${FUSION_MODE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --image_size ${IMAGE_SIZE} \
    --latent_dim ${LATENT_DIM} \
    --top_k ${TOP_K} \
    --temperature ${TEMPERATURE} \
    --device cuda \
    --seed 42"


if [ "$FREEZE_NCCT" = true ]; then
    CMD="$CMD --freeze_ncct"
fi


eval $CMD
