import json

from datasets import load_dataset

dataset = dataset = load_dataset("zhangirazerbayev/ocwcourses")

data = dataset["test"]

# 데이터 항목에서 'problems' 키를 'question' 키로 변경
for item in data:
    item["question"] = item.pop("problem")

with open("ocw_course.jsonl", "w") as jsonl_file:
    for item in data:
        json.dump(item, jsonl_file)
        jsonl_file.write("\n")
