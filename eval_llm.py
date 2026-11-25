"""LLM BD eval — perplexity on held-out Bangla text + one-shot generation.

Usage:
    python eval_llm.py --checkpoint out/tiny_demo.pth --prompt "ঢাকা বাংলাদেশের"
    python eval_llm.py --checkpoint out/tiny_demo.pth --perplexity
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import LLM  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402


def perplexity(model, tok, path, seq_len=128, device="cpu", limit=20):
    model.eval()
    total_nll, total_tok = 0.0, 0
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(l).get("text", "") for l in fh][:limit]
    with torch.no_grad():
        for text in lines:
            ids = tok.encode(text)
            for i in range(0, max(1, len(ids) - 1), seq_len):
                chunk = ids[i:i + seq_len + 1]
                if len(chunk) < 2:
                    continue
                x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
                _, loss = model(x, labels=y)
                total_nll += loss.item() * (len(chunk) - 1)
                total_tok += len(chunk) - 1
    return math.exp(total_nll / max(1, total_tok)) if total_tok else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/tiny_demo.pth")
    ap.add_argument("--prompt", default="ঢাকা বাংলাদেশের")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--perplexity", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train first:  python trainer/train_pretrain.py --tiny --steps 100")
        sys.exit(1)
    tok = BanglaTokenizer("model/tokenizer.json")
    model = LLM.load(args.checkpoint, device=args.device)
    if args.perplexity:
        ppl = perplexity(model, tok, "dataset/pretrain_val.jsonl", device=args.device)
        print(f"perplexity (held-out Bangla): {ppl:.2f}")
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=args.max_new, temperature=args.temperature)
    print("PROMPT:", args.prompt)
    print("OUTPUT:", tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
