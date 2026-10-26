import json
from io import BytesIO
from pathlib import Path

from datasets import Dataset, Image, load_dataset

jsonl_path = Path("Modality-Bias/data/convexity.jsonl")
new_dataset_path = Path("dataset_with_image/convexity")

ds = load_dataset("isobench/IsoBench", "math_convexity", split="validation")


def normalize(text):
    # Collapse all whitespace so spacing differences don't break comparisons.
    return " ".join(str(text).split())


def short_id(isobench_id):
    # "isobench/.../convexity_66" -> "convexity_66"
    return str(isobench_id).split("/")[-1]


def image_bytes(sample):
    buffer = BytesIO()
    sample["image"].save(buffer, format="PNG")
    return buffer.getvalue()


columns = ds.column_names

# Lookups store row indices; images are only encoded for matched rows.
rows_by_id = {}
rows_by_latex = {}
for i, sample in enumerate(ds.remove_columns("image")):
    if "id" in sample:
        rows_by_id.setdefault(short_id(sample["id"]), []).append(i)
    rows_by_latex.setdefault(normalize(sample["latex"]), []).append(i)

combined_records = []
problems = []
match_counts = {"id": 0, "latex": 0}

with jsonl_path.open(encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        # 1) Exact id match; 2) otherwise the ORIGINAL LaTeX expression.
        candidates, method = rows_by_id.get(record["id"], []), "id"
        if len(candidates) != 1:
            candidates, method = rows_by_latex.get(normalize(record["latex"]), []), "latex"

        if len(candidates) != 1:
            problems.append(f"{record['id']}: {len(candidates)} IsoBench rows matched")
            continue

        sample = ds[candidates[0]]

        # Sanity checks: a correct match agrees on expression and label.
        mismatches = []
        if normalize(sample["latex"]) != normalize(record["latex"]):
            mismatches.append("latex")
        if "code" in columns and normalize(sample["code"]) != normalize(record["code"]):
            mismatches.append("code")
        if str(sample["label"]).strip().lower() != str(record["label"]).strip().lower():
            mismatches.append("label")
        if mismatches:
            problems.append(f"{record['id']}: matched by {method} but {', '.join(mismatches)} differ")
            continue

        record["image"] = {"bytes": image_bytes(sample), "path": None}
        combined_records.append(record)
        match_counts[method] += 1

print(f"Matched {len(combined_records)} records: {match_counts}")
if problems:
    print(f"{len(problems)} records not matched:")
    for p in problems:
        print("  " + p)
    raise SystemExit("Not saving: fix the unmatched records first.")

combined_dataset = Dataset.from_list(combined_records)
combined_dataset = combined_dataset.cast_column("image", Image())
new_dataset_path.parent.mkdir(parents=True, exist_ok=True)
combined_dataset.save_to_disk(str(new_dataset_path))

print(f"Saved {len(combined_dataset)} records with images to {new_dataset_path}")