#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/input.jsonl /path/to/output.jsonl"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2

python ./predict.py \
    --test_file $INPUT_PATH \
    --output_file $OUTPUT_PATH \
    --model_name_or_path ./model \
    --text_column maintext \
    --summary_column title \
    --max_source_length 512 \
    --max_target_length 64 \
    --pad_to_max_length \
    --per_device_test_batch_size 4 \
    --strategy beam_search \
    --num_beams 20

