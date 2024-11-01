# ADL 2024 HW3 README

## Description

![ADL 2024 HW3 Description](./image/image.png)

將文言文跟中文互相翻譯

## Install

```bash
pyenv install 3.10.11
pyenv local 3.10.11

poetry env use 3.10.11
poetry install
poetry shell
```

## File Structure

- `data/`: 存放資料集
- `output/`: 存放模型
- `train.py`: 訓練模型
- `predict.py`: 預測結果
- `ppl.py`: 評估預測結果
- `ppl2.py`: 評估 zero-shot 跟 few-shot 的預測結果
- `plot.py`: 繪製學習曲線
- `utils.py`: 輔助函數

## Train

```bash
python train.py --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo --train_dataset_size 8000 --dataset ./data/train.json --source_max_len 360 --target_max_len 256 --bf16 --output_dir ./output/1 --per_device_train_batch_size 3 --gradient_accumulation_steps 4 --max_steps 500 --save_steps 50
```

## Evaluate

```bash
python ppl.py --base_model_path zake7749/gemma-2-2b-it-chinese-kyara-dpo --peft_path ./output/1/checkpoint-350 --test_data_path ./data/public_test.json
```

## Evaluate (zero-shot & few-shot)

```bash
python ppl2.py --base_model_path zake7749/gemma-2-2b-it-chinese-kyara-dpo --test_data_path ./data/public_test.json
```

## Predict

```bash
python predict.py --base_model_path zake7749/gemma-2-2b-it-chinese-kyara-dpo --peft_path ./output/1/checkpoint-350 --test_data_path ./data/private_test.json --output_path ./prediction.json
```

## Plot

```bash
python plot.py --base_model_path zake7749/gemma-2-2b-it-chinese-kyara-dpo --peft_path output/1 --test_data_path ./data/public_test.json --output_file ./output/1/learning_curve.png
```

## Script

```bash
bash download.sh
```

```bash
bash run.sh zake7749/gemma-2-2b-it-chinese-kyara-dpo adapter_checkpoint ./data/private_test.json ./prediction.json
```