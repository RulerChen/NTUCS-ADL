import argparse
import csv
import json
import logging
import os
from itertools import chain

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
)
from utils_qa import postprocess_qa_predictions

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Question Answering task"
    )

    parser.add_argument(
        "--context_file",
        type=str,
        default="./hw1/data/context.json",
        help="A csv or a json file containing the context data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./hw1/data/test.json",
        help="A csv or a json file containing the Prediction data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model dir.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Where to store the final model file.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--selection_model_name_or_path",
        type=str,
        help="Path to pretrained selection model.",
    )
    parser.add_argument(
        "--extra_model_name_or_path",
        type=str,
        help="Path to pretrained extra model.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    accelerator = Accelerator(
        **accelerator_log_kwargs,
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load the test dataset
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    test_raw_datasets = load_dataset(extension, data_files=data_files)

    # Load the selection model
    config = AutoConfig.from_pretrained(
        args.selection_model_name_or_path,
        cache_dir=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.selection_model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        args.selection_model_name_or_path,
        from_tf=bool(".ckpt" in args.selection_model_name_or_path),
        config=config,
        cache_dir=None,
    )

    # Resize model embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # padding
    padding = "max_length" if args.pad_to_max_length else False

    with open(args.context_file, "r", encoding="utf-8") as f:
        context = json.load(f)

    # Preprocessing
    def preprocess_function(examples):
        first_sentences = [[question] * 4 for question in examples["question"]]
        second_sentences = [
            [context[p] for p in paragraph] for paragraph in examples["paragraphs"]
        ]
        # labels = [
        #     examples["paragraphs"][i].index(paragraph)
        #     for i, paragraph in enumerate(examples["relevant"])
        # ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {
            k: [v[i : i + 4] for i in range(0, len(v), 4)]
            for k, v in tokenized_examples.items()
        }
        return tokenized_inputs

    with accelerator.main_process_first():
        test_datasets = test_raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=test_raw_datasets["test"].column_names,
        )

    # Data collator
    data_collator = default_data_collator
    test_dataloader = DataLoader(
        test_datasets,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=1,
    )

    # Prediction loop
    device = accelerator.device
    model.to(device)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.eval()

    with open(args.test_file, "r", encoding="utf-8") as f:
        test = json.load(f)

    outputs = []
    for i, input in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(**input)
        logits = output.logits.argmax(dim=1).item()
        outputs.append(
            {
                "id": test[i]["id"],
                "question": test[i]["question"],
                "context": context[test[i]["paragraphs"][logits]],
            }
        )

    # extra stage

    config = AutoConfig.from_pretrained(
        args.extra_model_name_or_path,
        cache_dir=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.extra_model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.extra_model_name_or_path,
        from_tf=bool(".ckpt" in args.extra_model_name_or_path),
        config=config,
        cache_dir=None,
    )

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    raw_datasets = datasets.Dataset.from_list(outputs)

    # Validation preprocessing
    def prepare_test_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    with accelerator.main_process_first():
        predict_dataset = raw_datasets.map(
            prepare_test_features,
            batched=True,
            remove_columns=raw_datasets.column_names,
            desc="Running tokenizer on prediction dataset",
        )

    predict_dataset_for_model = predict_dataset.remove_columns(
        ["example_id", "offset_mapping"]
    )
    predict_dataloader = DataLoader(
        predict_dataset_for_model,
        collate_fn=data_collator,
        batch_size=1,
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        return formatted_predictions

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    model.to(device)
    model, predict_dataloader = accelerator.prepare(model, predict_dataloader)

    all_start_logits = []
    all_end_logits = []

    model.eval()

    for step, batch in enumerate(predict_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if (
                not args.pad_to_max_length
            ):  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(
                    start_logits, dim=1, pad_index=-100
                )
                end_logits = accelerator.pad_across_processes(
                    end_logits, dim=1, pad_index=-100
                )

            all_start_logits.append(
                accelerator.gather_for_metrics(start_logits).cpu().numpy()
            )
            all_end_logits.append(
                accelerator.gather_for_metrics(end_logits).cpu().numpy()
            )

    max_len = max(
        [x.shape[1] for x in all_start_logits]
    )  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(
        all_start_logits, predict_dataset, max_len
    )
    end_logits_concat = create_and_fill_np_array(
        all_end_logits, predict_dataset, max_len
    )

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    predictions = post_processing_function(raw_datasets, predict_dataset, outputs_numpy)

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "answer"])
        for prediction in predictions:
            writer.writerow([prediction["id"], prediction["prediction_text"]])


if __name__ == "__main__":
    main()
