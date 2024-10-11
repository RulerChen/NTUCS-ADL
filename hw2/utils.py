import json


def jsonl_to_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def data_to_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return
