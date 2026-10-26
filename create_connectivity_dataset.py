import json
from io import BytesIO
from pathlib import Path

from datasets import Dataset, Image, load_dataset

jsonl_path = Path("Modality-Bias/data/connectivity.jsonl")
new_dataset_path = Path("dataset_with_image/connectivity")

ds = load_dataset("isobench/IsoBench", "graph_connectivity", split="validation")


def normalize_matrix(matrix):
    # Same graph -> same string, whether stored as a list or a string.
    if isinstance(matrix, str):
        matrix = json.loads(matrix)
    return json.dumps(matrix, separators=(",", ":"))


def short_id(isobench_id):
    # "isobench/.../connectivity_000" -> "connectivity_000"
    return str(isobench_id).split("/")[-1]


def image_bytes(sample):
    buffer = BytesIO()
    sample["image"].save(buffer, format="PNG")
    return buffer.getvalue()


# Lookups store row indices; images are only encoded for matched rows.
rows_by_id = {}
rows_by_content = {}  # (matrix, node_1, node_2) -> [row indices]
for i, sample in enumerate(ds.remove_columns("image")):
    if "id" in sample:
        rows_by_id.setdefault(short_id(sample["id"]), []).append(i)
    key = (normalize_matrix(sample["adjacency_matrix"]),
           int(sample["query_node_1"]), int(sample["query_node_2"]))
    rows_by_content.setdefault(key, []).append(i)

combined_records = []
problems = []
match_counts = {"id": 0, "matrix+nodes": 0}

with jsonl_path.open(encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        # 1) Exact id match; 2) otherwise the ORIGINAL matrix plus the target nodes.
        candidates, method = rows_by_id.get(record["id"], []), "id"
        if len(candidates) != 1:
            key = (normalize_matrix(record["adjacency_matrix"]),
                   int(record["query_node_1"]), int(record["query_node_2"]))
            candidates, method = rows_by_content.get(key, []), "matrix+nodes"

        if len(candidates) != 1:
            problems.append(f"{record['id']}: {len(candidates)} IsoBench rows matched")
            continue

        sample = ds[candidates[0]]

        # Sanity checks: a correct match agrees on graph, target nodes and label.
        mismatches = []
        if normalize_matrix(sample["adjacency_matrix"]) != normalize_matrix(record["adjacency_matrix"]):
            mismatches.append("adjacency_matrix")
        if (int(sample["query_node_1"]), int(sample["query_node_2"])) != (
                int(record["query_node_1"]), int(record["query_node_2"])):
            mismatches.append("query nodes")
        if bool(sample["label"]) != bool(record["label"]):
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