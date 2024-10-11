import argparse
import logging
import os

import datasets
import nltk
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from filelock import FileLock
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from transformers.utils import is_offline_mode

from utils import data_to_jsonl, jsonl_to_data

logger = get_logger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the test data.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )

    parser.add_argument(
        "--output_file", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="The number of highest probability vocabulary tokens to keep for top-k sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to module the next token probabilities.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy_search",
        help="The decoding strategy to use.",
        choices=[
            "beam_search",
            "top_k_sampling",
            "top_p_sampling",
            "temperature_sampling",
            "greedy_search",
        ],
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    data = jsonl_to_data(args.test_file)
    raw_datasets = datasets.Dataset.from_list(data)

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    column_names = raw_datasets.column_names
    ids = raw_datasets["id"]

    text_column = args.text_column

    padding = "max_length" if args.pad_to_max_length else False
    prefix = args.source_prefix if args.source_prefix is not None else ""

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, padding=padding, truncation=True
        )

        return model_inputs

    with accelerator.main_process_first():
        test_dataset = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_test_batch_size,
    )

    strategies = {
        "beam_search": {
            "num_beams": args.num_beams,
            "max_length": args.max_target_length,
        },
        "top_k_sampling": {
            "top_k": args.top_k,
            "max_length": args.max_target_length,
            "do_sample": True,
        },
        "top_p_sampling": {
            "top_p": args.top_p,
            "max_length": args.max_target_length,
            "do_sample": True,
        },
        "temperature_sampling": {
            "temperature": args.temperature,
            "max_length": args.max_target_length,
            "do_sample": True,
        },
        "greedy_search": {
            "max_length": args.max_target_length,
        },
    }

    gen_kwargs = strategies[args.strategy]

    results = []
    cnt = 0

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.eval()

    logger.info("***** Running Prediction *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {args.per_device_test_batch_size}")

    progress_bar = tqdm(
        range(len(test_dataloader)), disable=not accelerator.is_local_main_process
    )
    progress_bar.update(0)

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        generated_tokens = generated_tokens.cpu().numpy()

        decoded_preds = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        progress_bar.update(1)

        for pred in decoded_preds:
            results.append({"title": pred, "id": ids[cnt]})
            cnt += 1

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.output_file is not None:
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            data_to_jsonl(results, args.output_file)


if __name__ == "__main__":
    main()
