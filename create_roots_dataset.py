import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render without a display window
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, Image

jsonl_path = Path("Modality-Bias/data/roots.jsonl")
new_dataset_path = Path("dataset_with_image/roots")


def plot_polynomial(coefficients, roots):
    """Plot the ORIGINAL polynomial with its roots marked in red and labelled.

    Mixed Signals did not release its roots images or plotting code, so this
    follows the style of the paper's Figure 1: blue curve, red root markers
    with their values, grid and legend. Document these choices in the report.
    """
    roots = [float(r) for r in roots]
    lo, hi = min(roots), max(roots)
    pad = max(2.0, 0.15 * (hi - lo))
    x = np.linspace(lo - pad, hi + pad, 800)
    # coefficients are highest degree first, e.g. [3.0, 4.2] -> 3.0x + 4.2
    y = np.polyval([float(c) for c in coefficients], x)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.plot(x, y, label="Polynomial curve")
    ax.scatter(roots, [0] * len(roots), color="red", zorder=3, label="Roots")
    for r in roots:
        ax.annotate(f"{r:.1f}", (r, 0), textcoords="offset points",
                    xytext=(0, 6), ha="center", color="blue", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="PNG")
    plt.close(fig)
    return buffer.getvalue()


combined_records = []
problems = []

with jsonl_path.open(encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        # Sanity check: the coefficients must actually have these roots.
        coefficients = [float(c) for c in record["coefficients"]]
        scale = max(1.0, max(abs(c) for c in coefficients))
        residual = max(abs(np.polyval(coefficients, float(r))) for r in record["roots"])
        if residual > 1e-3 * scale:
            problems.append(f"{record['id']}: coefficients don't match roots (residual {residual:.3g})")
            continue

        # The image always shows the ORIGINAL polynomial (coefficients / roots),
        # never the _modify version, which only goes into the text input.
        record["image"] = {"bytes": plot_polynomial(record["coefficients"], record["roots"]), "path": None}
        combined_records.append(record)

print(f"Generated images for {len(combined_records)} records")
if problems:
    print(f"{len(problems)} records skipped:")
    for p in problems:
        print("  " + p)

combined_dataset = Dataset.from_list(combined_records)
combined_dataset = combined_dataset.cast_column("image", Image())
new_dataset_path.parent.mkdir(parents=True, exist_ok=True)
combined_dataset.save_to_disk(str(new_dataset_path))

print(f"Saved {len(combined_dataset)} records with images to {new_dataset_path}")