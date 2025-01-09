#!/bin/bash

# Define your list of argument values
#arg_values=("value1" "value2" "value3")
arg_values=(
    "datasets_realvideo/Samsung_dataset/full_res/apo_car_thinline"  # OOM at 1/2 scale
    "datasets_realvideo/Samsung_dataset/full_res/Dubai-CityofGold_Highfrequency1_2k"
    "datasets_realvideo/Samsung_dataset/full_res/Dubai-CityofGold_Highfrequency2_2k"
    "datasets_realvideo/Samsung_dataset/full_res/Impossible_Saves"
    "datasets_realvideo/Samsung_dataset/full_res/Patrick_Cantlay_shoots"
)

# Loop through each value in the list
for arg2 in "${arg_values[@]}"; do
    # Run your command with the new argument
    #my_command arg1 "$arg2"
    python -u evaluate.py --model "checkpoints/FlowDiffuser-things.pth"\
    --dataset infer_singleclip_ensemble --output_path "results/modelzoo-AF-T-ensemble/Samsung"\
    --scale_factor -1.0 --clip_root "$arg2" --ensembles 10
done
