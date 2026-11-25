"""Train the LLM BD Bangla BPE tokenizer on the sample corpus.

Usage:
    python train_tokenizer.py
    python train_tokenizer.py --vocab 8192 --out model/tokenizer.json
"""
import argparse
import json
import os

from model.tokenizer import train_bpe_tokenizer


def corpus_to_text():
    os.makedirs("dataset/.tmp", exist_ok=True)
    out = "dataset/.tmp/corpus.txt"
    with open(out, "w", encoding="utf-8") as fh:
        for path in ("dataset/pretrain_bangla.jsonl", "dataset/sft_bangla.jsonl"):
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as src:
                for line in src:
                    d = json.loads(line)
                    text = d.get("text") or " ".join(
                        str(d.get(k, "")) for k in ("instruction", "input", "output"))
                    if text.strip():
                        fh.write(text.strip() + "\n")
    return [out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--out", default="model/tokenizer.json")
    args = ap.parse_args()
    files = corpus_to_text()
    tok = train_bpe_tokenizer(files, vocab_size=args.vocab, save_path=args.out)
    print(f"tokenizer saved: {args.out}  vocab={tok.get_vocab_size()}")
    for t in ["আমি বাংলায় কথা বলি।", "ঢাকা বাংলাদেশের রাজধানী।"]:
        ids = tok.encode(t).ids
        print(f"  {t!r} -> {ids} -> {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
