#!/bin/bash
#mkdir -p checkpoints
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  python -u train.py --name fd-chairs --stage chairs --validation chairs --gpus 0 1 2 3 4 5 --num_steps 100000 --batch_size 12 --lr 0.00045 --image_size 384 512 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  python -u train.py --name fd-things --stage things --validation sintel --restore_ckpt checkpoints/fd-chairs.pth --gpus 0 1 2 3 4 5 --num_steps 200000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  python -u train.py --name fd-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/fd-things.pth --gpus 0 1 2 3 4 5 --num_steps 180000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.00001 --gamma=0.85
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  python -u train.py --name fd-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/fd-sintel.pth --gpus 0 1 2 3 4 5 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85

#CUDA_VISIBLE_DEVICES=4,5  python -u train.py --name fd-aft-sintel --stage sintel --validation sintel kitti --restore_ckpt checkpoints/FlowDiffuser-things.pth --gpus 0 1 --num_steps 180000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision
#CUDA_VISIBLE_DEVICES=4,5  python -u train.py --name fd-aft-sintel --stage sintel --validation sintel \
#--restore_ckpt checkpoints/FlowDiffuser-things.pth --gpus 0 1 \
#--num_steps 270000 --batch_size 4 --lr 0.000175 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
CUDA_VISIBLE_DEVICES=4,5  python -u train.py --name fd-aft-sintel2 --stage sintel --validation sintel kitti \
--restore_ckpt checkpoints/FlowDiffuser-things.pth --gpus 0 1 \
--num_steps 20 --batch_size 4 --lr 0.000175 --image_size 368 952 --wdecay 0.00001 --gamma=0.85 --mixed_precision