"""
Run one checkpoint on the Mixed Signals examples for one condition, with vLLM.

Same inputs, settings and output format as run.py (the Hugging Face version):

    python run_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen7b_base_vllm --condition reasoning

Kept identical to run.py, by importing them from it:
    GEN_CONFIG    max_new_tokens per condition
    IMAGE_CONFIG  min_pixels / max_pixels
    chat_prompt   the chat template: one user message, image then text,
                  no custom system prompt

Decoding is greedy, as in run.py. Hugging Face's greedy generate() still applies
the checkpoint's generation_config.json for `repetition_penalty` and its stop
tokens, so this script reads the same file and passes the same values to vLLM.

Output: outputs/<name>/<condition>.jsonl, the same fields as run.py.
Re-running the same command resumes: uids already in the output are skipped.
Do not mix backends in one output file; give vLLM runs their own --name.
"""
import argparse
import json
import time
from pathlib import Path

from transformers import AutoProcessor, GenerationConfig

from build import TASKS, build_examples
from run import GEN_CONFIG, IMAGE_CONFIG, chat_prompt

# Settings in a checkpoint's generation_config.json that would change greedy
# decoding. repetition_penalty is mirrored; the others are not supported here,
# so the script stops if a checkpoint sets them, rather than silently differing.
UNSUPPORTED_GREEDY_SETTINGS = ("no_repeat_ngram_size", "bad_words_ids", "suppress_tokens",
                               "begin_suppress_tokens", "forced_bos_token_id",
                               "forced_eos_token_id", "sequence_bias")


def greedy_settings_from_generation_config(model_path):
    """The parts of generation_config.json that Hugging Face applies under greedy decoding."""
    try:
        gc = GenerationConfig.from_pretrained(model_path)
    except OSError:
        gc = GenerationConfig()
    for name in UNSUPPORTED_GREEDY_SETTINGS:
        if getattr(gc, name, None):
            raise SystemExit(f"{model_path} sets `{name}` in generation_config.json; "
                             "run_vllm.py does not reproduce it. Use run.py for this checkpoint.")
    eos = gc.eos_token_id
    eos = [] if eos is None else (list(eos) if isinstance(eos, (list, tuple)) else [eos])
    return {
        "repetition_penalty": float(gc.repetition_penalty or 1.0),
        "stop_token_ids": [int(t) for t in eos],
    }


def read_done_uids(out_path):
    """uids already written. A half-written last line (a killed session) is removed."""
    done, good, bad = set(), [], 0
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["uid"])
                good.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError):
                bad += 1
    if bad:
        out_path.write_text("".join(good))
        print(f"Removed {bad} incomplete line(s) from {out_path}; those examples will be re-run.")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    ap.add_argument("--name", required=True, help="short name used for the output folder")
    ap.add_argument("--condition", required=True, choices=list(GEN_CONFIG))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--data", default=None, help="dataset_with_image folder (default: build.py's)")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--chunk_size", type=int, default=256,
                    help="prompts handed to vLLM per call; results are saved after each chunk. "
                         "vLLM schedules its own batches inside a chunk.")
    ap.add_argument("--max_model_len", type=int, default=8192,
                    help="prompt + reply token budget per sequence")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples per task (testing)")
    ap.add_argument("--dry_run", action="store_true", help="build examples and stop, no model")
    args = ap.parse_args()

    # 1) Examples for this condition: every task, both variants (same as run.py).
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
    done = read_done_uids(out_path)
    todo = [ex for ex in examples if ex.uid not in done]
    print(f"{len(examples)} examples, {len(done)} already done, {len(todo)} to run -> {out_path}")

    if args.dry_run or not todo:
        if todo:
            print("\n--- first user_text ---\n" + todo[0].user_text)
        return

    import vllm
    from vllm import LLM, SamplingParams

    greedy = greedy_settings_from_generation_config(args.model)
    cfg = GEN_CONFIG[args.condition]

    # 3) Record the exact settings once per run.
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "backend": "vllm", "vllm": vllm.__version__,
        "model": args.model, "name": args.name, "condition": args.condition,
        "tasks": args.tasks, "dtype": args.dtype, "chunk_size": args.chunk_size,
        "max_model_len": args.max_model_len,
        "gen_config": cfg, "decoding": {"temperature": 0.0, **greedy},
        "image_config": IMAGE_CONFIG,
    }, indent=2))

    # The Hugging Face processor renders the prompt text, exactly as in run.py.
    processor = AutoProcessor.from_pretrained(args.model, **IMAGE_CONFIG)

    print(f"Loading {args.model} with vLLM ...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs=dict(IMAGE_CONFIG),  # same image budget as run.py
        seed=0,
    )

    # 4) Generate task by task, in chunks; save after each chunk.
    start = time.time()
    n_done = 0
    with out_path.open("a") as f:
        for task in args.tasks:
            task_examples = [ex for ex in todo if ex.task == task]
            max_new = cfg["max_new_tokens_by_task"].get(task, cfg["max_new_tokens"])
            params = SamplingParams(
                temperature=0.0,          # greedy, as in run.py
                max_tokens=max_new,
                repetition_penalty=greedy["repetition_penalty"],
                stop_token_ids=greedy["stop_token_ids"] or None,
                skip_special_tokens=True,
            )
            for i in range(0, len(task_examples), args.chunk_size):
                chunk = task_examples[i:i + args.chunk_size]
                prompts = [chat_prompt(processor, ex.user_text) for ex in chunk]
                requests = [{"prompt": p, "multi_modal_data": {"image": ex.image.convert("RGB")}}
                            for p, ex in zip(prompts, chunk)]
                outputs = llm.generate(requests, params, use_tqdm=False)

                for ex, prompt, out in zip(chunk, prompts, outputs):
                    gen = out.outputs[0]
                    f.write(json.dumps({
                        "uid": ex.uid,
                        "task": ex.task,
                        "sample_id": ex.sample_id,
                        "variant": ex.variant,
                        "condition": ex.condition,
                        "model": args.name,
                        "prompt": prompt,
                        "response": gen.text.strip(),
                        "num_new_tokens": len(gen.token_ids),
                        "hit_max_new_tokens": gen.finish_reason == "length",
                    }) + "\n")
                f.flush()  # keep progress if the job is killed
                n_done += len(chunk)
                rate = n_done / (time.time() - start)
                print(f"[{task}] {n_done}/{len(todo)}  ({rate:.2f} ex/s)", flush=True)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()