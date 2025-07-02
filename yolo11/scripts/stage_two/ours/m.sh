#!/bin/bash

echo "Please add your generated_weights_path"

WBITS=(8 4 4)
ABITS=(8 4 8)

for ((i = 0; i < ${#WBITS[@]}; i++)); do
    model="yolo11m.pt"
    wbit="${WBITS[$i]}"
    abit="${ABITS[$i]}"
    echo "running $model ours in W$wbit A$abit"
    TRAIN_CONFIG="./external/config/kd.json" python -m external.main \
        --device cuda \
        --model $model \
        --model_quantize_mode.weight_bits "$wbit" \
        --model_quantize_mode.activation_bits "$abit" \
        --generated_weights_path
done

