"""LLM BD benchmark — held-out perplexity + demo-prompt generation.

Loads a checkpoint, measures perplexity on dataset/pretrain_val.jsonl,
runs every prompt in examples/demo_prompts_bn.md, and writes a markdown
table to examples/benchmark_results.md.

Usage:
    python scripts/benchmark.py --checkpoint out/tiny_demo.pth
    python scripts/benchmark.py --checkpoint out/sft_demo.pth --max-new 48
"""
import argparse
import math
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_llm import perplexity  # noqa: E402
from model.model import LLM, count_parameters  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402


def load_prompts(path="examples/demo_prompts_bn.md"):
    prompts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"\s*\d+\.\s+(.+?)\s*$", line)
            if m:
                prompts.append(m.group(1))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/tiny_demo.pth")
    ap.add_argument("--val", default="dataset/pretrain_val.jsonl")
    ap.add_argument("--prompts", default="examples/demo_prompts_bn.md")
    ap.add_argument("--out", default="examples/benchmark_results.md")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    torch.manual_seed(args.seed)
    tok = BanglaTokenizer("model/tokenizer.json")
    model = LLM.load(args.checkpoint, device=args.device)
    model.eval()
    n_params = count_parameters(model)

    t0 = time.time()
    ppl = perplexity(model, tok, args.val, device=args.device)
    prompts = load_prompts(args.prompts)
    rows = []
    for i, p in enumerate(prompts, 1):
        torch.manual_seed(args.seed + i)
        ids = torch.tensor([tok.encode(p)], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new,
                                 temperature=args.temperature)
        text = tok.decode(out[0].tolist())
        decoded_prompt = tok.decode(ids[0].tolist())
        if text.startswith(decoded_prompt):
            text = text[len(decoded_prompt):]
        reply = " ".join(text.strip().split())
        rows.append((i, p, reply))
    elapsed = time.time() - t0

    lines = [
        "# LLM BD — Benchmark Results (measured, not promised)",
        "",
        f"- **Checkpoint:** `{args.checkpoint}` ({n_params/1e6:.1f}M params)",
        f"- **Held-out perplexity** (`{args.val}`, 20 lines): **{ppl:.2f}**",
        f"- **Generation:** temperature={args.temperature}, top_k=40, "
        f"max_new_tokens={args.max_new}, seed={args.seed}+i per prompt",
        f"- **Prompts:** {len(prompts)} from `{args.prompts}`",
        f"- **Runtime:** {elapsed:.0f}s on CPU",
        "",
        "These are raw outputs from the tiny demo model — rough, repetitive, "
        "sometimes empty. That is the honest baseline: the pipeline works "
        "end-to-end, and fluency is what the full `llm-bd-64M` GPU run + more "
        "Bangla data are for.",
        "",
        "| # | Prompt | Model output |",
        "|---|---|---|",
    ]
    for i, p, r in rows:
        safe = (r or "(empty)").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {p} | {safe} |")
    lines.append("")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"perplexity: {ppl:.2f}  prompts: {len(prompts)}  wrote {args.out}")


if __name__ == "__main__":
    main()
