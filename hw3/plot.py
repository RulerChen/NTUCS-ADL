import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_bnb_config, get_prompt


def perplexity(
    model,
    tokenizer,
    data,
    max_length=2048,
):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions[
            "input_ids"
        ][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = (
            instruction_input_ids + output_input_ids
        )
        tokenized_instructions["attention_mask"][i] = [1] * len(
            tokenized_instructions["input_ids"][i]
        )
        output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length]
        )
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length]
        )
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_label) * shift_output_mask
            ).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
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
        default="",
        required=True,
        help="Path to test data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to the output file.",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load model
    bnb_config = get_bnb_config()

    checkpoint_list = []
    step_list = []
    for dir in os.listdir(args.peft_path):
        if os.path.isdir(os.path.join(args.peft_path, dir)):
            checkpoint_list.append(dir)
            step_list.append(int(dir.split("-")[-1]))
    checkpoint_list.sort(key=lambda x: int(x.split("-")[-1]))
    step_list.sort()

    print("Checkpoint list:")
    for i, checkpoint in enumerate(checkpoint_list):
        print(f"{i}: {checkpoint}")

    ppl_list = []

    for i, checkpoint in enumerate(checkpoint_list):
        if args.base_model_path:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load LoRA
        model = PeftModel.from_pretrained(model, args.peft_path + "/" + checkpoint)

        with open(args.test_data_path, "r") as f:
            data = json.load(f)
        model.eval()

        ppl = perplexity(model, tokenizer, data)
        print("Mean perplexity:", ppl["mean_perplexity"])
        ppl_list.append(ppl["mean_perplexity"])

    plt.figure(figsize=(10, 6))
    plt.plot(
        step_list,
        ppl_list,
        label="ppl score",
        color="blue",
        marker="o",
    )

    for x, y in zip(step_list, ppl_list):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom")

    plt.title("Learning curve")
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.legend()

    plt.savefig(args.output_file)
    plt.close()
