#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 /path/to/context.json /path/to/test.json /path/to/output/prediction.csv"
    exit 1
fi

CONTEXT_PATH=$1
TEST_PATH=$2
PREDICTION_PATH=$3

python ./predict.py \
    --output_file $PREDICTION_PATH \
    --context_file $CONTEXT_PATH \
    --test_file $TEST_PATH \
    --max_seq_length 512 \
    --pad_to_max_length \
    --selection_model_name_or_path ./select \
    --extra_model_name_or_path ./span \
    --max_answer_length 40 \
    --n_best_size 40 \
