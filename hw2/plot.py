import argparse
import json

import matplotlib.pyplot as plt
from tw_rouge import get_rouge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a summarization task"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./predict",
        help="The input data directory. Should contain the .jsonl files for the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predict",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    res = []

    result = {}
    with open("./data/public.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            result[line["id"]] = line["title"].strip() + "\n"

    result1 = {}
    with open(f"{args.input_dir}/1.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            result1[line["id"]] = line["title"].strip() + "\n"

    keys = result.keys()
    refs = [result[key] for key in keys]
    preds = [result1[key] for key in keys]
    res.append(get_rouge(preds, refs))

    result2 = {}
    with open(f"{args.input_dir}/2.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            result2[line["id"]] = line["title"].strip() + "\n"

    keys = result.keys()
    refs = [result[key] for key in keys]
    preds = [result2[key] for key in keys]
    res.append(get_rouge(preds, refs))

    result3 = {}
    with open(f"{args.input_dir}/4.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            result3[line["id"]] = line["title"].strip() + "\n"

    keys = result.keys()
    refs = [result[key] for key in keys]
    preds = [result3[key] for key in keys]
    res.append(get_rouge(preds, refs))

    result4 = {}
    with open(f"{args.input_dir}/6.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            result4[line["id"]] = line["title"].strip() + "\n"

    keys = result.keys()
    refs = [result[key] for key in keys]
    preds = [result4[key] for key in keys]
    res.append(get_rouge(preds, refs))

    result5 = {}
    with open(f"{args.input_dir}/8.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            result5[line["id"]] = line["title"].strip() + "\n"

    keys = result.keys()
    refs = [result[key] for key in keys]
    preds = [result5[key] for key in keys]
    res.append(get_rouge(preds, refs))

    result6 = {}
    with open(f"{args.input_dir}/10.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            result6[line["id"]] = line["title"].strip() + "\n"

    keys = result.keys()
    refs = [result[key] for key in keys]
    preds = [result6[key] for key in keys]
    res.append(get_rouge(preds, refs))

    epochs = [1, 2, 4, 6, 8, 10]
    rouge1 = [r["rouge-1"]["f"] * 100 for r in res]
    rouge2 = [r["rouge-2"]["f"] * 100 for r in res]
    rougeL = [r["rouge-l"]["f"] * 100 for r in res]

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        rouge1,
        label="rouge1",
        color="blue",
        marker="o",
    )
    plt.plot(
        epochs,
        rouge2,
        label="rouge2",
        color="red",
        marker="o",
    )
    plt.plot(
        epochs,
        rougeL,
        label="rougeL",
        color="green",
        marker="o",
    )
    plt.title("Rouge score")
    plt.xlabel("Epoch")
    plt.ylabel("Rouge score")
    plt.legend()

    plt.savefig(f"{args.output_dir}/rouge.png")
    plt.close()

    # plt.figure(figsize=(10, 6))

    # plt.plot(
    #     step,
    #     train_loss,
    #     label="loss",
    #     color="blue",
    #     marker="o",
    # )

    # plt.title("Loss")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    # plt.legend()

    # plt.savefig(f"{args.output_dir}/loss.png")
    # plt.close()


if __name__ == "__main__":
    main()
