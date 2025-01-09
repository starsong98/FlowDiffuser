#!/bin/bash
#mkdir -p checkpoints
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 12 --lr 0.00045 --image_size 384 512 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-things --stage things --validation sintel --restore_ckpt checkpoints/fd-chairs.pth --gpus 0 1 --num_steps 200000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/fd-things.pth --gpus 0 1 --num_steps 180000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.00001 --gamma=0.85
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/fd-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85

#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 12 --lr 0.00045 --image_size 384 512 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=0,1  python -u train.py --name fd-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/FlowDiffuser-things.pth --gpus 0 1 --num_steps 180000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.00001 --gamma=0.85

#CUDA_VISIBLE_DEVICES=4,5  python -u train.py --name fdnocascade-chairs --stage chairs --validation chairs --gpus 0 1 \
#--num_steps 100000 --batch_size 12 --lr 0.00045 --image_size 368 496 --wdecay 0.0001 --val_freq 5000 --model_type flowdiffuser_nocascade
#CUDA_VISIBLE_DEVICES=4,5  python -u train.py --name fdnocascade-chairs-debug --stage chairs --validation chairs --gpus 0 1 \
#--num_steps 100 --batch_size 12 --lr 0.00045 --image_size 368 496 --wdecay 0.0001 --val_freq 50 --model_type flowdiffuser_nocascade

# AutoFlow stage; all hyperparams same as Chairs stage as per author email - about 16~17 hours estimated on s177
#CUDA_VISIBLE_DEVICES=0,1  python -u train_resumable.py --name fd_nosp4_6iter-AF384x512 --stage autoflow --validation chairs \
#--gpus 0 1 --num_steps 100000 --val_freq 5000 --batch_size 6 --effective_batch_size 12 --lr 0.00045 \
#--image_size 384 512 --wdecay 0.0001 --iters 6 --sampling_timesteps 4 \
#--model_type flowdiffuser_nocascade_adjustable
# AutoFlow stage; all hyperparams same as Chairs stage as per author email - about 16~17 hours estimated on s177; resuming
CUDA_VISIBLE_DEVICES=0,1  python -u train_resumable.py --name fd_nosp4_6iter-AF384x512 --stage autoflow --validation chairs \
--restore_ckpt checkpoints/fd_nosp4_6iter-AF384x512/epoch_1_fd_nosp4_6iter-AF384x512_resumable.pth --resume_training \
--gpus 0 1 --num_steps 100000 --val_freq 5000 --batch_size 6 --effective_batch_size 12 --lr 0.00045 \
--image_size 384 512 --wdecay 0.0001 --iters 6 --sampling_timesteps 4 \
--model_type flowdiffuser_nocascade_adjustable

# FlyingThings stage; all hyperparams same as Things stage of full model
CUDA_VISIBLE_DEVICES=0,1  python -u train_resumable.py --name fd-nosp4_6iter-AF-T432x960 --stage things --validation sintel --restore_ckpt checkpoints/fd-chairs.pth \ #TODO change checkpoint
--gpus 0 1 --num_steps 200000 --val_freq 5000 --batch_size 2 --effective_batch_size 6 --lr 0.000175 \
--image_size 432 960 --wdecay 0.0001 --iters 6 --sampling_timesteps 4 \
--model_type flowdiffuser_nocascade_adjustable