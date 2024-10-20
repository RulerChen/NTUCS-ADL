from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    """Format the instruction as a prompt for LLM."""
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    return f"你是一個對文言文和中文非常了解的人，以下是你跟用戶之間的對話，你要對用戶的問題提供精準並且有用的回答。 USER: {instruction} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    """Get the BitsAndBytesConfig."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
