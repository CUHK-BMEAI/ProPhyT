#!/bin/bash



export LD_LIBRARY_PATH=/home/danying/miniconda3/envs/brain/lib:$LD_LIBRARY_PATH


WORK_DIR="workdir"
RUN_NAME="baseline_sam_0131"
DATA_PATH="data_penumbra_noblank_withvalid"


EPOCHS=15
BATCH_SIZE=2
LR=1e-4
IMAGE_SIZE=256



python train.py \
    --work_dir ${WORK_DIR} \
    --run_name ${RUN_NAME} \
    --data_path ${DATA_PATH} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --image_size ${IMAGE_SIZE} \
    --device cuda

