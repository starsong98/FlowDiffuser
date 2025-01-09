#!/bin/bash

# Define your list of argument values
#arg_values=("value1" "value2" "value3")
arg_values=(
    #"datasets_realvideo/XVFI/Longer_testset/Type1/TEST01_003_f0433"
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST05_158_f0321"
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST02_045_f0465"
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST03_081_f4833"
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST04_140_f3889"
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST06_001_f0273"
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST07_076_f1889"
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST08_079_f0321"
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST09_112_f0177"
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST10_172_f1905"
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST11_078_f4977"
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST12_087_f2721"
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST13_133_f4593"
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST14_146_f1761"
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST15_148_f0465"
)

# Loop through each value in the list
for arg2 in "${arg_values[@]}"; do
    # Run your command with the new argument
    #my_command arg1 "$arg2"
    python -u evaluate.py --model "checkpoints/FlowDiffuser-things.pth"\
    --dataset infer_singleclip_ensemble --output_path "results/modelzoo-AF-T-ensemble"\
    --scale_factor -2.0 --clip_root "$arg2" --ensembles 10
done
