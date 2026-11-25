"""LLM BD interactive Bangla CLI chat.

Loads ./out/sft_demo.pth, else ./out/tiny_demo.pth, else ./checkpoints/tiny/*.pth.
If no checkpoint exists it prints clear training instructions — it never fakes
generation.  Type 'exit' / 'quit' to leave.

Usage:
    python chat.py
    python chat.py --checkpoint out/sft_demo.pth --temperature 0.7
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import LLM  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402

CANDIDATES = ["out/sft_demo.pth", "out/tiny_demo.pth"]


def find_checkpoint(explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    hits = sorted(glob.glob("checkpoints/tiny/*.pth"))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    ckpt = find_checkpoint(args.checkpoint)
    if ckpt is None:
        print("LLM BD — কোনো ট্রেন করা মডেল পাওয়া যায়নি।")
        print("আগে ট্রেন করুন (CPU-তে কয়েক মিনিট):")
        print("  python train_tokenizer.py")
        print("  python trainer/train_pretrain.py --tiny --steps 100")
        print("  python trainer/train_sft.py --init out/tiny_demo.pth --steps 60")
        print("  python chat.py")
        sys.exit(1)
    if not os.path.exists("model/tokenizer.json"):
        print("Tokenizer not found — run: python train_tokenizer.py")
        sys.exit(1)
    tok = BanglaTokenizer("model/tokenizer.json")
    model = LLM.load(ckpt, device="cpu")
    print(f"LLM BD চ্যাট — মডেল: {ckpt}  (বের হতে 'exit' লিখুন)")
    while True:
        try:
            user = input("\nআপনি: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nবিদায়!")
            break
        if user.lower() in ("exit", "quit", "বিদায়"):
            print("বিদায়!")
            break
        if not user:
            continue
        prompt = tok.encode_chat(user)
        ids = torch.tensor([tok.encode(prompt, add_special=False)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=args.max_new,
                             temperature=args.temperature)
        reply = tok.decode(out[0].tolist())
        # strip the echoed prompt (decode() skips special tokens, so compare
        # against the decoded prompt, not the raw template string)
        decoded_prompt = tok.decode(ids[0].tolist())
        if reply.startswith(decoded_prompt):
            reply = reply[len(decoded_prompt):]
        print("LLM BD:", reply.strip() or "(খালি উত্তর — আরও ট্রেনিং দরকার)")


if __name__ == "__main__":
    main()
