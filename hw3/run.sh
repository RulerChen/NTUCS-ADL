#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 /path/to/model-folder /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json"
    exit 1
fi

model_folder=$1
adapter_checkpoint=$2
input_json=$3
output_json=$4

python predict.py \
    --base_model_path $model_folder \
    --peft_path $adapter_checkpoint \
    --test_data_path $input_json \
    --output_path $output_json

