"""
Score Mixed Signals outputs: parse each reply, then compute the bias metrics.

    python score.py                         # every outputs/<name>/<condition>.jsonl
    python score.py --models qwen7b_base qwen7b_papo

Reads outputs written by run.py and the answer key from build.py (joined on
task + sample_id). Writes to scores/:
    summary.csv     one row per model x condition x task (+ "all" rows)
    complexity.csv  same metrics per complexity bucket
    parsed.csv      every reply with its parsed answer and category (for inspection)

Metric definitions (per model x condition x task):
    aligned_acc        % of aligned replies equal to A
    filtered_n         samples correct on aligned, excluding A == A'
    image/text/both/other/unparseable_pct
                       categories of the MISMATCHED reply on the filtered samples
    B                  image_pct - text_pct  (the paper's bias value)
    common_n, common_B B on the samples every compared model got right on aligned
    unfiltered_*       image/text % over all mismatched replies (A != A'), no filter
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from build import DATA_DIR, TASKS, build_answer_key

ROOT_TOL = 0.05  # roots are given to one decimal; half a step separates neighbours


# ================================================================ parsing
ANSWER_RE = re.compile(r"answer\s*[:：]\s*(.*)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(response):
    """Text after the LAST 'Answer:'; falls back to the last non-empty line.

    Returns (answer_text, source) with source "tag" or "last_line".
    """
    matches = list(ANSWER_RE.finditer(response))
    if matches:
        text, source = matches[-1].group(1), "tag"
    else:
        lines = [l for l in response.strip().splitlines() if l.strip()]
        text, source = (lines[-1] if lines else ""), "last_line"
    text = text.strip().splitlines()[0] if text.strip() else ""
    # Remove decoration the model may copy from the format spec or markdown.
    text = text.replace("**", "").strip().strip("<>").strip().rstrip(".").strip()
    return text, source


def _one_of(text, options):
    """The single option word present in text, or None if zero or several are present."""
    found = {o for o in options if re.search(rf"\b{o}\b", text, re.IGNORECASE)}
    return found.pop() if len(found) == 1 else None


def parse(task, answer_text, key_entry):
    """Normalise an answer string to the type used in the answer key, or None."""
    if not answer_text:
        return None
    if task in ("connectivity", "vsr"):
        word = _one_of(answer_text, ["true", "false"])
        return None if word is None else word == "true"
    if task == "convexity":
        return _one_of(answer_text, ["convex", "concave"])
    if task == "roots":
        nums = [float(x) for x in NUMBER_RE.findall(answer_text)]
        return sorted(nums) if nums else None
    if task == "science":
        choices = key_entry["choices"]
        # A letter at the start: "B", "B.", "(B)", "B) neither ...".
        # The \b stops "Solution A" being read as the letter S.
        m = re.match(r"^\(?([A-Za-z])\b", answer_text)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < len(choices):
                return idx
        # Otherwise the model may have written the choice text itself.
        norm = lambda s: " ".join(str(s).lower().split())
        for i, c in enumerate(choices):
            if norm(answer_text) == norm(c):
                return i
        return None
    raise KeyError(task)


def _roots_cover(parsed, target):
    """Every root in target has a distinct match in parsed (within tolerance)."""
    pool = list(parsed)
    for t in target:
        hit = next((p for p in pool if abs(p - t) <= ROOT_TOL), None)
        if hit is None:
            return False
        pool.remove(hit)
    return True


def equals(task, parsed, answer):
    if parsed is None:
        return False
    if task == "roots":
        return len(parsed) == len(answer) and _roots_cover(parsed, answer)
    return parsed == answer


def categorize(task, parsed, a, a_prime):
    """Mismatched reply -> image / text / both / other / unparseable."""
    if parsed is None:
        return "unparseable"
    if equals(task, parsed, a):
        return "image"
    if equals(task, parsed, a_prime):
        return "text"
    if task == "roots":
        # "Both": reports the image's roots and the text's roots together,
        # e.g. [-1.4, 5.2] when A = [-1.4] and A' = [5.2] (paper, Fig. 3d).
        union = sorted(set(a) | set(a_prime))
        if _roots_cover(parsed, a) and _roots_cover(parsed, a_prime) and len(parsed) <= len(union):
            return "both"
    return "other"


# ================================================================ loading
def load_outputs(out_dir, models=None):
    rows = []
    for path in sorted(Path(out_dir).glob("*/*.jsonl")):
        model, condition = path.parent.name, path.stem
        if models and model not in models:
            continue
        with path.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["model"], r["condition"] = model, condition
                    rows.append(r)
    if not rows:
        raise SystemExit(f"No outputs found under {out_dir}/<name>/<condition>.jsonl")
    return pd.DataFrame(rows)


def parse_all(df, key):
    records = []
    for r in df.itertuples(index=False):
        k = key.get((r.task, r.sample_id))
        if k is None:
            raise KeyError(f"{r.task}/{r.sample_id} is not in the answer key")
        answer_text, source = extract_answer(r.response)
        parsed = parse(r.task, answer_text, k)
        a, a_prime = k["answer_image"], k["answer_text"]
        rec = {
            "model": r.model, "condition": r.condition, "task": r.task,
            "sample_id": r.sample_id, "variant": r.variant,
            "answer_extracted": answer_text, "answer_source": source,
            "parsed": json.dumps(parsed), "A": json.dumps(a), "A_prime": json.dumps(a_prime),
            "a_equals_a_prime": k["a_equals_a_prime"], "complexity": k["complexity"],
            "unparseable": parsed is None,
            "hit_max_new_tokens": bool(getattr(r, "hit_max_new_tokens", False)),
        }
        if r.variant == "aligned":
            rec["correct"] = equals(r.task, parsed, a)
            rec["category"] = None
        else:
            rec["correct"] = None
            rec["category"] = categorize(r.task, parsed, a, a_prime)
        records.append(rec)
    return pd.DataFrame(records)


def pair(parsed):
    """One row per (model, condition, task, sample): aligned result + mismatched category."""
    al = parsed[parsed.variant == "aligned"][
        ["model", "condition", "task", "sample_id", "correct", "unparseable",
         "a_equals_a_prime", "complexity", "hit_max_new_tokens"]
    ].rename(columns={"unparseable": "aligned_unparseable", "hit_max_new_tokens": "aligned_hit_max"})
    mm = parsed[parsed.variant == "mismatched"][
        ["model", "condition", "task", "sample_id", "category", "hit_max_new_tokens"]
    ].rename(columns={"hit_max_new_tokens": "mismatched_hit_max"})
    keys = ["model", "condition", "task", "sample_id"]
    p = al.merge(mm, on=keys, how="outer", indicator=True)
    incomplete = p[p["_merge"] != "both"]
    if len(incomplete):
        print(f"WARNING: {len(incomplete)} samples lack one of the two variants "
              f"(run still in progress?) and are excluded.")
    p = p[p["_merge"] == "both"].drop(columns="_merge")
    p["filtered"] = p["correct"].astype(bool) & ~p["a_equals_a_prime"].astype(bool)
    return p


# ================================================================ metrics
CATS = ["image", "text", "both", "other", "unparseable"]


def pct(x, n):
    return round(100.0 * x / n, 2) if n else None


def metrics(g, common_ids=None):
    n = len(g)
    usable = g[~g.a_equals_a_prime.astype(bool)]
    f = g[g.filtered]
    out = {
        "n_samples": n,
        "aligned_acc": pct(g.correct.astype(bool).sum(), n),
        "aligned_unparseable_pct": pct(g.aligned_unparseable.astype(bool).sum(), n),
        "filtered_n": len(f),
    }
    for c in CATS:
        out[f"{c}_pct"] = pct((f.category == c).sum(), len(f))
    out["B"] = (round(out["image_pct"] - out["text_pct"], 2)
                if out["image_pct"] is not None else None)

    if common_ids is not None:
        cg = f[f.apply(lambda r: (r.task, r.sample_id) in common_ids, axis=1)]
        out["common_n"] = len(cg)
        ci, ct = pct((cg.category == "image").sum(), len(cg)), pct((cg.category == "text").sum(), len(cg))
        out["common_B"] = round(ci - ct, 2) if ci is not None else None

    out["unfiltered_n"] = len(usable)
    out["unfiltered_image_pct"] = pct((usable.category == "image").sum(), len(usable))
    out["unfiltered_text_pct"] = pct((usable.category == "text").sum(), len(usable))
    out["mismatched_unparseable_pct"] = pct((g.category == "unparseable").sum(), n)
    hit = g.aligned_hit_max.astype(bool).sum() + g.mismatched_hit_max.astype(bool).sum()
    out["hit_max_tokens_pct"] = pct(hit, 2 * n)
    return out


def common_samples(p):
    """Per condition: samples that EVERY model got right on aligned (and A != A')."""
    common = {}
    for cond, g in p.groupby("condition"):
        n_models = g.model.nunique()
        ok = g[g.filtered].groupby(["task", "sample_id"]).model.nunique()
        common[cond] = set(ok[ok == n_models].index)
    return common


def complexity_bucket(task, value):
    if task == "convexity":  # paper bins expression length in steps of 10 characters
        lo = int(value) // 10 * 10
        return f"{lo}-{lo + 9} chars"
    if task == "connectivity":
        return f"{int(value)} edges"
    if task == "roots":
        return f"degree {int(value)}"
    return str(value)  # science difficulty (E/M), vsr relation


def summarize(p):
    common = common_samples(p)
    rows, cx_rows = [], []
    for (model, cond), g in p.groupby(["model", "condition"]):
        ids = common[cond]
        for task, gt in g.groupby("task"):
            rows.append({"model": model, "condition": cond, "task": task, **metrics(gt, ids)})
            gt = gt.assign(bucket=[complexity_bucket(task, v) for v in gt.complexity])
            for bucket, gb in gt.groupby("bucket"):
                m = metrics(gb)
                cx_rows.append({"model": model, "condition": cond, "task": task, "bucket": bucket,
                                **{k: m[k] for k in ["n_samples", "aligned_acc", "filtered_n",
                                                     "image_pct", "text_pct", "both_pct",
                                                     "other_pct", "unparseable_pct", "B"]}})
        rows.append({"model": model, "condition": cond, "task": "all", **metrics(g, ids)})
    order = {t: i for i, t in enumerate(list(TASKS) + ["all"])}
    summary = pd.DataFrame(rows).sort_values(
        ["model", "condition", "task"], key=lambda s: s.map(order) if s.name == "task" else s)
    return summary, pd.DataFrame(cx_rows)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these --name values")
    ap.add_argument("--scores", default="scores")
    args = ap.parse_args()

    df = load_outputs(args.outputs, args.models)
    key = build_answer_key(sorted(df.task.unique()), args.data)
    parsed = parse_all(df, key)
    p = pair(parsed)
    summary, complexity = summarize(p)

    out = Path(args.scores)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    complexity.to_csv(out / "complexity.csv", index=False)
    parsed.to_csv(out / "parsed.csv", index=False)

    cols = ["model", "condition", "task", "aligned_acc", "filtered_n",
            "image_pct", "text_pct", "other_pct", "unparseable_pct", "B", "common_B",
            "mismatched_unparseable_pct", "hit_max_tokens_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[cols].to_string(index=False))
    print(f"\nWrote {out}/summary.csv, complexity.csv, parsed.csv")
    fallback = (parsed.answer_source == "last_line").mean() * 100
    if fallback:
        print(f"Note: {fallback:.1f}% of replies had no 'Answer:' tag; their last line was used.")


if __name__ == "__main__":
    main()