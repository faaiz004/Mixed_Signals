"""
Run one checkpoint on the Mixed Signals examples for one condition, with
Hugging Face transformers.

    python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen7b_base --condition direct
    python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen7b_base --condition reasoning
    python run.py --model /path/to/rl_checkpoint      --name qwen7b_papo --condition direct

Inputs you choose: --model, --name, --condition (and optionally --tasks, --batch_size).
Fixed per condition so every checkpoint is evaluated identically: decoding
settings, max_new_tokens, image resolution (GEN_CONFIG / IMAGE_CONFIG below).
Both variants (aligned + mismatched) are always run.

Output: outputs/<name>/<condition>.jsonl, one line per example with the raw reply.
The answer key stays in build.py's Example; score.py joins on `uid`.
Re-running the same command resumes: uids already in the output are skipped.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from build import TASKS, build_examples

# ---------------------------------------------------------------- fixed settings
# Same for every checkpoint.
GEN_CONFIG = {
    "direct": {
        "max_new_tokens": 64,
        "max_new_tokens_by_task": {},
    },
    "reasoning": {
        "max_new_tokens": 1024,
        "max_new_tokens_by_task": {},
    },
}
# Greedy decoding for every run.
DECODING = {"do_sample": False, "num_beams": 1}

# Qwen2.5-VL resizes each image to a pixel budget. Fixing it keeps memory
# predictable (COCO photos are large) and identical across checkpoints.
IMAGE_CONFIG = {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}


# ---------------------------------------------------------------- model
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path, device, attn_implementation):
    dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, **IMAGE_CONFIG)
    # Batched generation with a decoder-only model needs LEFT padding,
    # so every prompt ends right where generation starts.
    processor.tokenizer.padding_side = "left"
    return model, processor


def chat_prompt(processor, user_text):
    """The ONLY place the chat template is applied: image first, then text,
    no custom system prompt (the checkpoint's default template applies)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_batch(model, processor, examples, max_new_tokens):
    prompts = [chat_prompt(processor, ex.user_text) for ex in examples]
    images = [ex.image.convert("RGB") for ex in examples]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens, **DECODING)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]  # drop the prompt part
    replies = processor.batch_decode(new_tokens, skip_special_tokens=True)

    pad_id = processor.tokenizer.pad_token_id
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    results = []
    for prompt, reply, toks in zip(prompts, replies, new_tokens.tolist()):
        toks = [t for t in toks if t != pad_id]
        finished = bool(toks) and toks[-1] in eos_ids
        results.append({
            "prompt": prompt,
            "response": reply.strip(),
            "num_new_tokens": len(toks),
            "hit_max_new_tokens": (not finished) and len(toks) >= max_new_tokens,
        })
    return results


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    ap.add_argument("--name", required=True, help="short name used for the output folder")
    ap.add_argument("--condition", required=True, choices=list(GEN_CONFIG))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--data", default=None, help="dataset_with_image folder (default: build.py's)")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--attn", default="sdpa", help="sdpa (default) or flash_attention_2")
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples per task (testing)")
    ap.add_argument("--dry_run", action="store_true", help="build examples and stop, no model")
    args = ap.parse_args()

    # 1) Examples for this condition: every task, both variants.
    build_kwargs = {"tasks": args.tasks}
    if args.data:
        build_kwargs["data_dir"] = args.data
    examples = build_examples(args.condition, **build_kwargs)
    if args.limit:
        kept, seen = [], {}
        for ex in examples:
            if seen.get(ex.task, 0) < 2 * args.limit:  # 2 = both variants
                kept.append(ex)
                seen[ex.task] = seen.get(ex.task, 0) + 1
        examples = kept

    # 2) Resume: skip uids already written.
    out_path = Path(args.out_dir) / args.name / f"{args.condition}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {json.loads(line)["uid"] for line in f if line.strip()}
    todo = [ex for ex in examples if ex.uid not in done]
    print(f"{len(examples)} examples, {len(done)} already done, {len(todo)} to run -> {out_path}")

    if args.dry_run or not todo:
        if todo:
            print("\n--- first user_text ---\n" + todo[0].user_text)
        return

    # 3) Record the exact settings once per run.
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "model": args.model, "name": args.name, "condition": args.condition,
        "tasks": args.tasks, "batch_size": args.batch_size, "attn": args.attn,
        "gen_config": GEN_CONFIG[args.condition], "decoding": DECODING,
        "image_config": IMAGE_CONFIG, "torch": torch.__version__,
    }, indent=2))

    device = pick_device()
    print(f"Loading {args.model} on {device} ...")
    model, processor = load_model(args.model, device, args.attn)

    # 4) Generate task by task (similar prompt lengths -> less padding), in batches.
    cfg = GEN_CONFIG[args.condition]
    start = time.time()
    n_done = 0
    with out_path.open("a") as f:
        for task in args.tasks:
            task_examples = [ex for ex in todo if ex.task == task]
            max_new = cfg["max_new_tokens_by_task"].get(task, cfg["max_new_tokens"])
            for i in range(0, len(task_examples), args.batch_size):
                batch = task_examples[i:i + args.batch_size]
                for ex, res in zip(batch, generate_batch(model, processor, batch, max_new)):
                    f.write(json.dumps({
                        "uid": ex.uid,
                        "task": ex.task,
                        "sample_id": ex.sample_id,
                        "variant": ex.variant,
                        "condition": ex.condition,
                        "model": args.name,
                        **res,
                    }) + "\n")
                f.flush()  # keep progress if the job is killed
                n_done += len(batch)
                rate = n_done / (time.time() - start)
                print(f"[{task}] {n_done}/{len(todo)}  ({rate:.2f} ex/s)", flush=True)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()