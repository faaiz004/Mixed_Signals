"""
Turn the saved Mixed Signals datasets into model-ready examples.

    build_example(record, task, condition, variant) -> one Example
    build_examples(condition, tasks, data_dir)      -> every task, both variants, one condition

The flags only change the prompt text:
    variant   "aligned"    -> original text fields   (image and text agree: answer A)
              "mismatched" -> *_modify text fields   (image says A, text says A')
    condition "direct" / "reasoning" -> passed to prompts.render

The image, A, A', the A == A' flag and the complexity value come from the
record alone, so both variants of a sample carry identical answers.
"""
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from prompts import CONDITIONS, format_choices, render

TASKS = ("connectivity", "convexity", "roots", "science", "vsr")
VARIANTS = ("aligned", "mismatched")
DATA_DIR = Path("dataset_with_image")


@dataclass
class Example:
    uid: str                 # task/sample_id/variant/condition, unique per example
    task: str
    sample_id: str
    condition: str
    variant: str
    image: Any               # PIL image, always the ORIGINAL (shared by both variants)
    user_text: str           # prompt text from prompts.render
    answer_image: Any        # A:  answer implied by the image (and the original text)
    answer_text: Any         # A': answer implied by the modified text
    a_equals_a_prime: bool   # True for the samples where image/text answers coincide
    complexity: Any          # the paper's difficulty proxy for this task
    meta: dict = field(default_factory=dict)

    @property
    def gold(self):
        """Correct answer for this example: A when aligned. Undefined when mismatched."""
        return self.answer_image if self.variant == "aligned" else None


# ---------------------------------------------------------------- helpers
def _connected(adj, a, b):
    seen, stack = {a}, [a]
    while stack:
        u = stack.pop()
        for v, edge in enumerate(adj[u]):
            if edge and v not in seen:
                seen.add(v)
                stack.append(v)
    return b in seen


def _num_edges(adj):
    # Undirected graph: each edge appears twice in the symmetric matrix.
    return sum(sum(1 for e in row if e) for row in adj) // 2


def _same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    return a == b


def _pick(record, field_name, variant):
    """Original field for 'aligned', its _modify version for 'mismatched'."""
    return record[field_name if variant == "aligned" else f"{field_name}_modify"]


# ---------------------------------------------------------------- per task
# Each returns (fields for prompts.render, A, A', complexity).

def _connectivity(record, variant):
    adj = ast.literal_eval(record["adjacency_matrix"])
    adj_mod = ast.literal_eval(record["adjacency_matrix_modify"])
    n1, n2 = int(record["query_node_1"]), int(record["query_node_2"])
    fields = {
        "node_color": record["query_nodes_color"],
        "adjacency_matrix": _pick(record, "adjacency_matrix", variant),
        "node_1": n1,
        "node_2": n2,
    }
    a = bool(record["label"])
    a_prime = _connected(adj_mod, n1, n2)  # recomputed, not assumed to be "not label"
    return fields, a, a_prime, _num_edges(adj)


def _convexity(record, variant):
    # Use `code`, not `latex`: only `code` has a modified version.
    fields = {"expression": _pick(record, "code", variant)}
    a = record["label"]
    a_prime = "concave" if a == "convex" else "convex"
    return fields, a, a_prime, len(record["code"])


def _roots(record, variant):
    fields = {"polynomial": _pick(record, "polynomial", variant)}
    a = sorted(float(r) for r in record["roots"])
    a_prime = sorted(float(r) for r in record["roots_modify"])
    return fields, a, a_prime, int(record["degree"])


def _science(record, variant):
    choices = record["choices"]
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)  # stored as a Python-style list string
    fields = {
        "description": _pick(record, "description", variant),
        "question": record["question"],
        "choices": format_choices(choices),
    }
    a = int(record["label"])            # 0-based index; letter A = 0
    a_prime = int(record["label_modify"])
    return fields, a, a_prime, record["difficulty"]


def _vsr(record, variant):
    fields = {
        "paragraph": _pick(record, "extended_caption", variant),
        "statement": record["caption"],
    }
    # Every released row has label 1: the original paragraph supports the
    # statement, the modified paragraph contradicts it.
    a, a_prime = True, False
    return fields, a, a_prime, record["relation"]


TASK_BUILDERS = {
    "connectivity": _connectivity,
    "convexity": _convexity,
    "roots": _roots,
    "science": _science,
    "vsr": _vsr,
}


def _sample_id(record, task):
    # VSR has no id field; the image name is not unique (images are reused),
    # so combine it with the relation and caption index set by build_examples.
    return record["id"] if "id" in record else record["_vsr_id"]


# ---------------------------------------------------------------- public API
def build_example(record, task, condition, variant, image=None):
    """One sample -> one Example for the given condition and variant.

    `image` lets callers pass an already-decoded image so both variants share it.
    """
    if task not in TASK_BUILDERS:
        raise KeyError(f"Unknown task {task!r}")
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition {condition!r}")
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}")

    fields, a, a_prime, complexity = TASK_BUILDERS[task](record, variant)
    sample_id = _sample_id(record, task)
    return Example(
        uid=f"{task}/{sample_id}/{variant}/{condition}",
        task=task,
        sample_id=sample_id,
        condition=condition,
        variant=variant,
        image=image if image is not None else record["image"],
        user_text=render(task, condition, fields),
        answer_image=a,
        answer_text=a_prime,
        a_equals_a_prime=_same(a, a_prime),
        complexity=complexity,
    )


def load_task(task, data_dir=DATA_DIR):
    return load_from_disk(str(Path(data_dir) / task))


def build_examples(condition, tasks=TASKS, data_dir=DATA_DIR):
    """Every record of every task, BOTH variants, for ONE condition."""
    examples = []
    for task in tasks:
        ds = load_task(task, data_dir)
        for i in range(len(ds)):
            record = ds[i]                 # decodes the image once ...
            if task == "vsr":
                record["_vsr_id"] = f"vsr_{i:04d}"
            for variant in VARIANTS:       # ... and both variants share it
                examples.append(build_example(record, task, condition, variant,
                                              image=record["image"]))
    return examples


def build_answer_key(tasks=TASKS, data_dir=DATA_DIR):
    """(task, sample_id) -> answers and metadata, WITHOUT decoding images.

    Uses the same per-task logic and the same VSR numbering as build_examples,
    so score.py interprets replies exactly as the examples were built.
    """
    key = {}
    for task in tasks:
        ds = load_task(task, data_dir)
        if "image" in ds.column_names:
            ds = ds.remove_columns("image")
        for i in range(len(ds)):
            record = ds[i]
            if task == "vsr":
                record["_vsr_id"] = f"vsr_{i:04d}"
            _, a, a_prime, complexity = TASK_BUILDERS[task](record, "aligned")
            entry = {
                "answer_image": a,
                "answer_text": a_prime,
                "a_equals_a_prime": _same(a, a_prime),
                "complexity": complexity,
            }
            if task == "science":
                choices = record["choices"]
                entry["choices"] = ast.literal_eval(choices) if isinstance(choices, str) else list(choices)
            key[(task, _sample_id(record, task))] = entry
    return key


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--condition", default="reasoning", choices=CONDITIONS)
    args = ap.parse_args()

    examples = build_examples(args.condition, data_dir=args.data)
    uids = [e.uid for e in examples]
    per_task = Counter((e.task, e.variant) for e in examples)
    ambiguous = Counter(e.task for e in examples if e.a_equals_a_prime and e.variant == "aligned")

    print(f"{len(examples)} examples for condition={args.condition}")
    for task in TASKS:
        print(f"  {task:12s} aligned={per_task[(task, 'aligned')]:4d} "
              f"mismatched={per_task[(task, 'mismatched')]:4d} A==A'={ambiguous[task]}")
    print(f"unique uids: {len(set(uids)) == len(uids)}")
    print(f"total A==A' samples: {sum(ambiguous.values())} (expected 9)")

    # Print one aligned/mismatched pair per task: they should differ only in the modified text.
    for task in TASKS:
        pair = [e for e in examples if e.task == task][:2]
        print(f"\n===== {task}: {pair[0].sample_id}  A={pair[0].answer_image!r}  A'={pair[0].answer_text!r} =====")
        for e in pair:
            print(f"--- {e.variant} ---")
            print(e.user_text[-400:])