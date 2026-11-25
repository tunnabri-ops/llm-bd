"""LLM BD pretraining — from scratch, AdamW + cosine LR, checkpoint/resume.

Usage:
    python trainer/train_pretrain.py --tiny --steps 100
    python trainer/train_pretrain.py --config 64M --steps 2000 --device cuda
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import LLM, ModelConfig, count_parameters  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402


def load_tokens(path, tok, seq_len):
    ids = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if text:
                ids += tok.encode(text)
    return ids


def get_batch(ids, batch, seq_len, device):
    starts = [random.randint(0, len(ids) - seq_len - 1) for _ in range(batch)]
    x = torch.tensor([[ids[s + i] for i in range(seq_len)] for s in starts], dtype=torch.long)
    y = torch.tensor([[ids[s + i + 1] for i in range(seq_len)] for s in starts], dtype=torch.long)
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny", choices=["tiny", "64M"])
    ap.add_argument("--tiny", action="store_true", help="alias for --config tiny")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out/tiny_demo.pth")
    ap.add_argument("--from-resume", type=int, default=0)
    ap.add_argument("--weights-only", action="store_true",
                    help="save model weights only (no optimizer state) — much smaller file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.tiny:
        args.config = "tiny"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    tok = BanglaTokenizer("model/tokenizer.json")
    cfg = ModelConfig.llm_bd_tiny(tok.vocab_size) if args.config == "tiny" \
        else ModelConfig.llm_bd_64m(tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    model = LLM(cfg).to(device)
    print(f"LLM BD {args.config}: {count_parameters(model)/1e6:.1f}M params, device={device}")

    ids = load_tokens("dataset/pretrain_bangla.jsonl", tok, args.seq_len)
    print(f"corpus tokens: {len(ids):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    start_step = 0
    if args.from_resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck.get("step", 0)
        print(f"resumed from step {start_step}")

    def lr_at(step):
        warm = 10
        if step < warm:
            return args.lr * step / warm
        p = (step - warm) / max(1, args.steps - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    model.train()
    first_loss = last_loss = None
    t0 = time.time()
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(ids, args.batch, args.seq_len, device)
        _, loss = model(x, labels=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {loss.item():.4f} lr {lr_at(step):.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt = {"config": cfg.to_dict(), "state_dict": model.state_dict(), "step": args.steps}
    if not args.weights_only:
        ckpt["optimizer"] = opt.state_dict()
    torch.save(ckpt, args.out)
    print(f"saved {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"LOSS_START={first_loss:.4f} LOSS_END={last_loss:.4f} STEPS={args.steps}")


if __name__ == "__main__":
    main()
