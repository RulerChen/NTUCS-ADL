import argparse
import logging
import os
import json

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from datasets import Dataset
import torch
from tqdm import tqdm

from utils import get_bnb_config, get_prompt


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Tuning")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="zake7749/gemma-2-2b-it-chinese-kyara-dpo",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/public_test.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output/output.json",
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default="./output/checkpoint-100/"
    )
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    data = Dataset.from_json(args.input_file)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=get_bnb_config(),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    loraconifg = LoraConfig.from_pretrained(args.checkpoint_file)
    model = PeftModel(model, loraconifg)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    predictions = []
    for item in tqdm(data):
        instruction = item["instruction"]

        inputs = tokenizer(
            instruction,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device), max_length=args.max_length
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(
            {"id": item["id"], "instruction": instruction, "output": decoded_output}
        )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    logger.info(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
