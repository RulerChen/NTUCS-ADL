import torch
from transformers import BitsAndBytesConfig


def get_prompt(instruction: str) -> str:
    """Format the instruction as a prompt for LLM."""
    # LoRA
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

    # Zero-shot
    return f"你是一個對文言文和白話文非常了解的文學專家，以下是你跟用戶之間的對話，你要依據用戶的提示將內容翻譯成文言文或白話文。 用戶: {instruction} 文學專家: "

    # Few-shot
    # return (
    #     f"你是一個對文言文和白話文非常了解的文學專家，以下是你跟用戶之間的對話，你要依據用戶的提示將內容翻譯成文言文或白話文。 用戶: {instruction} 文學專家: "
    #     f"用戶: 翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案："
    #     f"文學專家: 雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
    #     f"用戶: 沒過十天，鮑泉果然被拘捕。\n幫我把這句話翻譯成文言文"
    #     f"文學專家: 後未旬，果見囚執。"
    #     f"用戶: 辛未，命吳堅為左丞相兼樞密使，常楙參知政事。\n把這句話翻譯成現代文。"
    #     f"文學專家: 初五，命令吳堅為左承相兼樞密使，常增為參知政事。"
    #     f"用戶: 十八年，奚、契丹侵犯邊界，以皇上為河北道元帥，信安王為副，率禦史大夫李朝隱、京兆尹裴亻由先等八總管討伐他們。\n翻譯成文言文："
    #     f"文學專家: 十八年，奚、契丹犯塞，以上為河北道元帥，信安王禕為副，帥禦史大夫李朝隱、京兆尹裴伷先等八總管兵以討之。"
    #     f"用戶: 正月，甲子朔，鼕至，太後享通天宮；赦天下，改元。\n把這句話翻譯成現代文。"
    #     f"文學專家: 聖曆元年正月，甲子朔，鼕至，太後在通天宮祭祀；大赦天下，更改年號。"
    #     f"用戶: {instruction} 文學專家: "
    # )


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
