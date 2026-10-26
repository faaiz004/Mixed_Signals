import json
from io import BytesIO
from pathlib import Path

from datasets import Dataset, Image, load_dataset

jsonl_path = Path("Modality-Bias/data/science.jsonl")
new_dataset_path = Path("dataset_with_image/science")

subsets = {
    "physics": load_dataset("isobench/IsoBench", "physics", split="validation"),
    "chemistry": load_dataset("isobench/IsoBench", "chemistry", split="validation"),
}


def normalize(text):
    # Collapse all whitespace so line breaks / double spaces don't break comparisons.
    return " ".join(str(text).split())


def short_id(isobench_id):
    # "isobench/science/chemistry_35" -> "chemistry_35"
    return str(isobench_id).split("/")[-1]


def image_bytes(sample):
    buffer = BytesIO()
    sample["image"].save(buffer, format="PNG")
    return buffer.getvalue()


# short id -> (subset name, row index)
rows_by_id = {}
for name, ds in subsets.items():
    for i, isobench_id in enumerate(ds["id"]):
        sid = short_id(isobench_id)
        if sid in rows_by_id:
            raise ValueError(f"Duplicate IsoBench id {sid}")
        rows_by_id[sid] = (name, i)

combined_records = []
problems = []

with jsonl_path.open(encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        if record["id"] not in rows_by_id:
            problems.append(f"{record['id']}: id not found in IsoBench")
            continue

        name, row = rows_by_id[record["id"]]
        sample = subsets[name][row]

        # Sanity checks: the matched row must have the same ORIGINAL text and label.
        mismatches = []
        if normalize(sample["question"]) != normalize(record["question"]):
            mismatches.append("question")
        if normalize(sample["description"]) != normalize(record["description"]):
            mismatches.append("description")
        if int(sample["label"]) != int(record["label"]):
            mismatches.append("label")
        if mismatches:
            problems.append(f"{record['id']}: matched by id but {', '.join(mismatches)} differ")
            continue

        record["image"] = {"bytes": image_bytes(sample), "path": None}
        combined_records.append(record)

print(f"Matched {len(combined_records)} records by id")
if problems:
    print(f"{len(problems)} records not matched:")
    for p in problems:
        print("  " + p)

combined_dataset = Dataset.from_list(combined_records)
combined_dataset = combined_dataset.cast_column("image", Image())
new_dataset_path.parent.mkdir(parents=True, exist_ok=True)
combined_dataset.save_to_disk(str(new_dataset_path))

print(f"Saved {len(combined_dataset)} records with images to {new_dataset_path}")