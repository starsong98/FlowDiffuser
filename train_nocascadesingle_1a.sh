#!/bin/bash
#mkdir -p checkpoints
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 12 --lr 0.00045 --image_size 384 512 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-things --stage things --validation sintel --restore_ckpt checkpoints/fd-chairs.pth --gpus 0 1 --num_steps 200000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/fd-things.pth --gpus 0 1 --num_steps 180000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.00001 --gamma=0.85
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/fd-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85

#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 12 --lr 0.00045 --image_size 384 512 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/FlowDiffuser-things.pth --gpus 0 1 --num_steps 180000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.00001 --gamma=0.85

CUDA_VISIBLE_DEVICES=4,5  python -u train.py --name fdnocascadesingle-chairs --stage chairs --validation chairs --gpus 0 1 \
--num_steps 200000 --batch_size 16 --lr 0.00045 --image_size 368 496 --wdecay 0.0001 --val_freq 5000 --model_type flowdiffuser_nocascade_single
# this uses about 29000 GiB of VRAM.
# could squeeze a little bit more...