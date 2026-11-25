"""LLM BD SFT — supervised fine-tuning with chat template, prompt tokens masked.

Usage:
    python trainer/train_sft.py --init out/tiny_demo.pth --steps 60
    python trainer/train_sft.py --init checkpoints/llm-bd-64M/pretrain.pth --steps 500
"""
import argparse
import json
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import LLM, ModelConfig  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402


def build_examples(path, tok, seq_len):
    exs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            prompt = tok.encode_chat(d["instruction"], d.get("input", ""))
            p_ids = tok.encode(prompt, add_special=False)
            a_ids = tok.encode(d["output"] + "<eos>", add_special=False)
            ids = (p_ids + a_ids)[:seq_len]
            labels = ([-100] * len(p_ids) + a_ids)[:seq_len]
            exs.append((ids, labels))
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="out/tiny_demo.pth")
    ap.add_argument("--data", default="dataset/sft_bangla.jsonl")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out/sft_demo.pth")
    ap.add_argument("--weights-only", action="store_true",
                    help="save model weights only (no optimizer state) — much smaller file")
    args = ap.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    tok = BanglaTokenizer("model/tokenizer.json")
    model = LLM.load(args.init, device=device) if os.path.exists(args.init) \
        else LLM(ModelConfig.llm_bd_tiny(tok.vocab_size)).to(device)
    model.train()
    exs = build_examples(args.data, tok, args.seq_len)
    print(f"SFT examples: {len(exs)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    first = last = None
    for step in range(args.steps):
        batch = random.sample(exs, min(args.batch, len(exs)))
        maxlen = max(len(i) for i, _ in batch)
        x = torch.full((len(batch), maxlen), 0, dtype=torch.long)
        y = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        for j, (ids, labels) in enumerate(batch):
            x[j, :len(ids)] = torch.tensor(ids)
            y[j, :len(labels)] = torch.tensor(labels)
        _, loss = model(x.to(device), labels=y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"sft step {step:4d} loss {loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt = {"config": model.cfg.to_dict(), "state_dict": model.state_dict(),
            "step": args.steps}
    if not args.weights_only:
        ckpt["optimizer"] = opt.state_dict()
    torch.save(ckpt, args.out)
    print(f"saved {args.out}  LOSS_START={first:.4f} LOSS_END={last:.4f}")


if __name__ == "__main__":
    main()
