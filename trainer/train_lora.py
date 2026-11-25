"""LLM BD LoRA — low-rank adaptation written from scratch (no peft dependency).

Freezes the base model and trains only small rank-r adapters on q/v projections.
Usage:
    python trainer/train_lora.py --init out/tiny_demo.pth --steps 40 --rank 8
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import LLM  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402
from trainer.train_sft import build_examples  # noqa: E402


class LoRALinear(nn.Module):
    """Wraps a frozen Linear with a trainable low-rank update: W + (B @ A) * scale."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def attach_lora(model, rank=8, targets=("q_proj", "v_proj")):
    n = 0
    for block in model.blocks:
        for name in targets:
            mod = getattr(block.attn, name)
            setattr(block.attn, name, LoRALinear(mod, rank=rank))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="out/tiny_demo.pth")
    ap.add_argument("--data", default="dataset/sft_bangla.jsonl")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out/lora_demo.pth")
    args = ap.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    tok = BanglaTokenizer("model/tokenizer.json")
    model = LLM.load(args.init, device=device)
    n = attach_lora(model, rank=args.rank)
    # LoRA means ONLY the adapters train — freeze everything else (the
    # embedding, MLP and untouched attention weights stay fixed).
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA adapters: {n}, trainable {trainable:,} / {total:,} params "
          f"({100*trainable/total:.2f}%)")
    model.train()
    exs = build_examples(args.data, tok, 256)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
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
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"lora step {step:4d} loss {loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    adapters = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"config": model.cfg.to_dict(), "lora_state_dict": adapters,
                "rank": args.rank, "step": args.steps}, args.out)
    print(f"saved LoRA adapters {args.out}  LOSS_START={first:.4f} LOSS_END={last:.4f}")


if __name__ == "__main__":
    main()
