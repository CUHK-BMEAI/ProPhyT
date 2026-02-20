#!/bin/bash




export LD_LIBRARY_PATH=/home/danying/miniconda3/envs/brain/lib:$LD_LIBRARY_PATH


WORK_DIR="workdir"
RUN_NAME="cpal_sam_dense_prompt_v2"
DATA_PATH="data_penumbra_noblank_withvalid"



MODEL_TYPE="vit_b"


CHECKPOINT="$WORK_DIR/models/$RUN_NAME/latest.pth"
STAGE1B_CKPT="scripts/logs/cpal_stage1b_v2/best_model.pth"
PROTOTYPE_BANK="scripts/logs/cpal_stage1a_post/ctp_prototypes_post.npy"
FUSION_MODE="dense_prompt"
LATENT_DIM=256
TOP_K=5
TEMPERATURE=0.1


IMAGE_SIZE=256
BOXES_PROMPT="True"
POINT_NUM=1          # 使用1个点
ITER_POINT=1         # 不使用迭代
SAVE_PRED="True"


python test_cpal_sam_v2.py \
    --work_dir $WORK_DIR \
    --run_name $RUN_NAME \
    --data_path $DATA_PATH \
    --model_type $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --stage1b_ckpt $STAGE1B_CKPT \
    --prototype_bank $PROTOTYPE_BANK \
    --fusion_mode $FUSION_MODE \
    --latent_dim $LATENT_DIM \
    --top_k $TOP_K \
    --temperature $TEMPERATURE \
    --image_size $IMAGE_SIZE \
    --boxes_prompt $BOXES_PROMPT \
    --point_num $POINT_NUM \
    --iter_point $ITER_POINT \
    --save_pred $SAVE_PRED \
    --device cuda
