#!/bin/sh
# sleep 6h
PARTITION=Segmentation

# GPU_ID=0,1,2,3
GPU_ID=0


CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=6224 train.py \
                        --epochs 50 \
                        --condition mask \
                        --lr 1e-4 \
                        --fold 0 \
                        --logpath trn1_coco_mask_fold0
