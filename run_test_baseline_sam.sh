#!/bin/bash

WORK_DIR="workdir"
RUN_NAME="baseline_sam"
DATA_PATH="data_penumbra_noblank_withvalid"

MODEL_TYPE="vit_b"
CHECKPOINT="$WORK_DIR/models/$RUN_NAME/best_dice_sam.pth"


IMAGE_SIZE=256
BOXES_PROMPT="True"
POINT_NUM=1         
ITER_POINT=8        #实际并未使用,因为用了box
SAVE_PRED="True"


python test.py \
    --work_dir $WORK_DIR \
    --run_name $RUN_NAME \
    --data_path $DATA_PATH \
    --model_type $MODEL_TYPE \
    --sam_checkpoint $CHECKPOINT \
    --image_size $IMAGE_SIZE \
    --boxes_prompt $BOXES_PROMPT \
    --point_num $POINT_NUM \
    --iter_point $ITER_POINT \
    --save_pred $SAVE_PRED \
    --device cuda
