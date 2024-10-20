import argparse
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import bitsandbytes as bnb
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
from packaging import version
from packaging.version import parse
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer, Seq2SeqTrainer,
                          set_seed)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils import get_bnb_config, get_prompt

logger = logging.getLogger(__name__)


# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = True


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Tuning")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="zake7749/gemma-2-2b-it-chinese-kyara-dpo",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--source_max_len",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--target_max_len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/train.json",
    )
    parser.add_argument(
        "--train_on_source",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--full_finetune",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--double_quant",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="nf4",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_32bit",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
    )
    parser.add_argument(
        "--remove_unused_columns",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--group_by_length",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    return args


def find_all_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        logger.info("CUDA is available.")
    else:
        logger.info("CUDA is not available.")

    if args.full_finetune:
        assert args.bits in [16, 32]
    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    print(f"loading base model {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
    )

    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    model.config.torch_dtype = compute_dtype

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,  # Fast tokenizer giving issues.
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    if not args.full_finetune:
        if checkpoint_dir is not None:
            logger.info("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(
                model, os.path.join(checkpoint_dir, "adapter_model"), is_trainable=True
            )
        else:
            logger.info(f"adding LoRA modules...")
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model, tokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        "input": [],
        "output": [],
    }
    for example_instances in examples["instances"]:
        for instance in example_instances:
            out["input"].append(instance["instruction_with_input"])
            out["output"].append(instance["output"])
    if extract_reformulations:
        for example_reformulations in examples["reformulations"]:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out["input"].append(instance["instruction_with_input"])
                    out["output"].append(instance["output"])
    return out


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def local_dataset(dataset_name):
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:

    dataset = local_dataset(args.dataset)
    dataset = dataset.map(
        lambda x: {
            "input": [get_prompt(ins) for ins in x["instruction"]],
            "output": x["output"],
        }
    )

    dataset = dataset["train"].train_test_split(
        train_size=args.train_size, shuffle=True, seed=args.seed
    )
    valid_dataset = dataset["test"]

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        valid_dataset=valid_dataset,
        data_collator=data_collator,
    )


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished

        max_step = 0
        for filename in os.path.listdir(checkpoint_dir):
            if os.path.isdir(
                os.path.join(checkpoint_dir, filename)
            ) and filename.startswith("checkpoint"):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint

        checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint-{max_step}")
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!

    return None, False  # first training


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.info("Detected that training was already completed!")

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    logger.info("loaded model")
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Verifying the datatypes and parameter counts before training.

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    main()
