import json

with open("./data/train.jsonl", "r", encoding="utf-8") as f:
    train_data = [json.loads(jline) for jline in f]


title_len = 100
maintext_len = 1024

title_cnt = 0
maintext_cnt = 0

for data in train_data:
    if len(data["title"]) > title_len:
        title_cnt += 1
    if len(data["maintext"]) > maintext_len:
        maintext_cnt += 1

print(f"Total data: {len(train_data)}")
print(f"Title length > {title_len}: {title_cnt}")
print(f"Maintext length > {maintext_len}: {maintext_cnt}")
