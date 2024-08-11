import jsonlines as jsl
from pprint import pprint

records = list(jsl.open("MATH-full.jsonl"))

# split records into 1000-length chunks
chunks = [records[i:i+1000] for i in range(0, len(records), 1000)]

# save with _pt1.jsonl suffix
for i, chunk in enumerate(chunks):
    with jsl.open(f"MATH-full_pt{i+1}.jsonl", "w") as f:
        f.write_all(chunk)