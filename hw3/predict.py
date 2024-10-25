import argparse
import json

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_bnb_config, get_prompt


def generate_predictions(model, tokenizer, data):
    predictions = []

    for item in tqdm(data):
        instruction = get_prompt(item["instruction"])
        input_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(
            model.device
        )

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, max_new_tokens=128)

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs = output_text[len(instruction) :].strip()

        predictions.append(
            {
                "id": item["id"],
                "output": outputs,
            }
        )

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat.",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.json",
        help="Path to save the prediction output.",
    )
    args = parser.parse_args()

    bnb_config = get_bnb_config()
    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    predictions = generate_predictions(model, tokenizer, data)

    with open(args.output_path, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to {args.output_path}")
