from pathlib import Path
import json

data_dir = Path("Modality-Bias/data")
domains = ["connectivity.jsonl", "convexity.jsonl", "roots.jsonl", "science.jsonl", "vsr.jsonl"]

for domain in domains:
    curr_jsonl = data_dir / domain
    with curr_jsonl.open(encoding="utf-8") as file:
        record = json.loads(file.readline())
        print(f"Keys in {domain}:")
        for key in record:
            print(f"    {key}")

