# Mixed Signals evaluation

This codebase measures whether vision-language models follow the **image** or the **text** when the
two disagree, using the Mixed Signals benchmark ([Pezeshkpour et al., 2025](https://arxiv.org/abs/2504.08974)).
It evaluates Qwen2.5-VL checkpoints, base and RL-trained, under a **direct** and a **reasoning**
prompting condition.

Each sample has an image and a text that agree (**aligned**, answer A) and a version where the text
is edited to imply a different answer (**mismatched**, answer A′) while the image is unchanged. The
model's answer on the mismatched version shows which input it relied on.

| Task | Samples | Question |
|---|---|---|
| Connectivity | 128 | Are two nodes in a graph connected? |
| Convexity | 256 | Is a function convex or concave? |
| Roots | 400 | What are the roots of a polynomial? |
| Science | 99 | Physics and chemistry multiple choice |
| VSR | 846 | Is a spatial statement about a photo true? |

## Layout

```
Mixed Signals/
├── Modality-Bias/                    # megagonlabs/Modality-Bias (jsonl data)
├── create_connectivity_dataset.py    # build image datasets
├── create_convexity_dataset.py
├── create_roots_dataset.py
├── create_science_dataset.py
├── create_vsr_dataset.py
├── inspecting_arrow_datasets.py      # inspect the saved datasets
├── investigate_json.py               # inspect the jsonl fields
├── dataset_with_image/               # datasets with images (Arrow)
├── mixed_signals_eval/
│   ├── prompts.py                    # prompt templates
│   ├── build.py                      # samples -> prompts, images and answers
│   ├── run.py                        # model generation
│   └── score.py                      # parsing and metrics
├── outputs/                          # model replies
└── scores/                           # metrics
```

Run all commands from `Mixed Signals/`.

## Setup

```bash
python -m venv cs689_venv && source cs689_venv/bin/activate
pip install datasets pillow matplotlib numpy pandas
pip install torch torchvision "transformers>=4.49" accelerate   # for generation

git clone https://github.com/megagonlabs/Modality-Bias.git
```

## 1. Build the datasets

Each script reads a task's jsonl from `Modality-Bias/data/`, attaches the image, and saves the result
to `dataset_with_image/<task>/`.

| Script | Image source |
|---|---|
| `create_connectivity_dataset.py` | IsoBench `graph_connectivity` (Hugging Face) |
| `create_convexity_dataset.py` | IsoBench `math_convexity` |
| `create_science_dataset.py` | IsoBench `physics` and `chemistry` |
| `create_vsr_dataset.py` | COCO 2017, from each row's `image_link` |
| `create_roots_dataset.py` | Generated: plot of the polynomial with its roots in red |

```bash
python create_connectivity_dataset.py
python create_convexity_dataset.py
python create_science_dataset.py
python create_vsr_dataset.py
python create_roots_dataset.py
```

Check the datasets:

```bash
python mixed_signals_eval/build.py
```

## 2. Generate

```bash
python mixed_signals_eval/run.py --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen7b_base --condition direct
python mixed_signals_eval/run.py --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen7b_base --condition reasoning
```

| Argument | Meaning |
|---|---|
| `--model` | Hugging Face id or local checkpoint path |
| `--name` | Label for the checkpoint; names the output folder |
| `--condition` | `direct` (answer only) or `reasoning` (explanation, then answer) |
| `--tasks` | Subset of tasks (default: all) |
| `--batch_size` | Default 8 |
| `--limit N` | First N samples per task |
| `--dry_run` | Build the prompts without loading a model |

Each run covers both variants of every sample (3,458 prompts) and resumes if re-run.
Decoding is greedy, with a limit of 64 new tokens (direct) or 1024 (reasoning).

Output: `outputs/<name>/<condition>.jsonl` (one reply per line) and `<condition>.meta.json` (run settings).

## 3. Score

```bash
python mixed_signals_eval/score.py
python mixed_signals_eval/score.py --models qwen7b_base qwen7b_papo
```

Writes `scores/summary.csv` (per model, condition and task), `scores/complexity.csv` (per difficulty
level) and `scores/parsed.csv` (every reply with its parsed answer).

| Metric | Meaning |
|---|---|
| `aligned_acc` | Accuracy when image and text agree |
| `filtered_n` | Samples answered correctly when aligned (the set bias is measured on) |
| `image_pct` | Mismatched answers that follow the image |
| `text_pct` | Mismatched answers that follow the text |
| `B` | `image_pct − text_pct`: positive favours the image, negative the text |
| `common_B` | B on the samples every compared model answered correctly when aligned |
| `unfiltered_image_pct`, `unfiltered_text_pct` | The same proportions over all samples |
| `both_pct`, `other_pct`, `unparseable_pct` | Remaining answer categories |
| `hit_max_tokens_pct` | Replies that reached the token limit |

## How it works

```
prompts.py ──► build.py ──► run.py ──► outputs/ ──► score.py ──► scores/
```

- `prompts.py` defines each task's prompt: instruction, output format, inputs. The two conditions
  differ only in the output format.
- `build.py` fills the prompt with the original text (aligned) or the edited text (mismatched),
  and attaches the image and the answers A and A′.
- `run.py` applies the model's chat template and generates the replies.
- `score.py` parses each reply's `Answer:` line, pairs the aligned and mismatched replies of each
  sample, and computes the metrics.
