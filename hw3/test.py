import json

data = []

with open("./data/public_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ins_thres = 324
output_thres = 256

ins_cnt = 0
output_cnt = 0

max_ins = 0
max_output = 0
for i in range(len(data)):
    if len(data[i]["instruction"]) > ins_thres:
        ins_cnt += 1
    if len(data[i]["output"]) > output_thres:
        output_cnt += 1
    max_ins = max(max_ins, len(data[i]["instruction"]))
    max_output = max(max_output, len(data[i]["output"]))

print("Instruction length >", ins_thres, ":", ins_cnt)
print("Output length >", output_thres, ":", output_cnt)
print("Max instruction length:", max_ins)
print("Max output length:", max_output)
