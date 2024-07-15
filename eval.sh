#!/bin/bash
#python evaluate.py --model=weights/FlowDiffuser-things.pth  --dataset=sintel
#python evaluate.py --model=weights/FlowDiffuser-things.pth  --dataset=kitti
python evaluate.py --model=checkpoints/FlowDiffuser-things.pth  --dataset=sintel --output_path="results/modelzoo-T"
#python evaluate.py --model=checkpoints/FlowDiffuser-things.pth  --dataset=kitti --output_path="results/modelzoo-T"
#python evaluate.py --model=checkpoints/FlowDiffuser-things.pth  --dataset=chairs --output_path="results/modelzoo-T"