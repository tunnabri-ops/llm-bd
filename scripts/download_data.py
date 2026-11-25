"""LLM BD data downloader.

Downloads extra Bangla text for pretraining (optional). Sources are public-domain /
CC-licensed Bangla corpora. No data is bundled — run this only if you want a
bigger corpus than the 200+ sentences in dataset/.

Usage:
    python scripts/download_data.py --out dataset/extra_bangla.txt
    python dataset/build_dataset.py --extra-text dataset/extra_bangla.txt
"""
import argparse
import os
import urllib.request

SOURCES = {
    # Bangla Wikipedia dump (first chunk) — CC BY-SA, public mirror
    "bn_wiki": "https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/extra_bangla.txt")
    ap.add_argument("--source", default="bn_wiki", choices=list(SOURCES))
    args = ap.parse_args()
    url = SOURCES[args.source]
    print(f"Downloading {args.source} from {url} ...")
    print("NOTE: the Wikipedia dump is large (~100MB+ compressed).")
    print("For the 2-hour run, the bundled dataset/pretrain_bangla.jsonl is enough;")
    print("use this only if you want to scale up. Press Ctrl+C to cancel.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, args.out + ".bz2")
    print(f"saved {args.out}.bz2 — decompress with: bzip2 -d {args.out}.bz2")
    print("Then clean it to one sentence per line and pass it to dataset/build_dataset.py")


if __name__ == "__main__":
    main()
