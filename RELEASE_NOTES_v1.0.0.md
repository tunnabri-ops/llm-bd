# LLM BD v1.0.0 — Release Notes

**First release: the tiny demo you can download and chat with today, plus
the full pipeline to train the real 64M model yourself.**

## What's in v1.0.0

- **Bigger Bangla dataset, all original text:** 1,016 pretrain sentences
  (was 207) across history, geography, science, maths, health, technology,
  AI, sports/cricket, literature, daily life, world news, environment and
  economics — plus 325 SFT pairs (was 132), including maths word problems
  with working, Bangla grammar, 30+ BN↔EN translations, summarisation,
  20+ Python coding examples, general knowledge, reasoning and creative
  writing. No duplicates; 20 held-out val lines that are *not* in train
  (the v1 builder accidentally leaked them in — fixed).
- **Retrained tiny demo** (`llm-bd-tiny`, ~4.3M params) on CPU, 200 steps.
- **New `scripts/benchmark.py`** — held-out perplexity + all 30 demo
  prompts, writing real results to `examples/benchmark_results.md`.
- **Bangla training guide** (`docs/TRAINING_GUIDE_BN.md`) and a viral
  launch kit (`LAUNCH.md`).
- **Bug fixes:** LoRA now really trains only the adapters (0.75% of
  params — v1 accidentally left 88% trainable); SFT can save weights-only
  checkpoints (~17 MB instead of 52 MB).

## Measured numbers (tiny demo, CPU)

| Metric | v1.0 baseline | v1.0.0 |
|---|---|---|
| Pretrain data | 207 sentences | **1,016 sentences** (54,332 tokens) |
| SFT data | 132 pairs | **325 pairs** |
| Pretrain loss (steps) | 6.3745 → 3.0293 (100) | **6.4114 → 2.2212 (200)** |
| SFT loss (steps) | 5.5351 → 2.6178 (30) | **6.6552 → 1.4696 (60)** |
| LoRA loss (steps) | 5.5351 → 3.0786 (10)¹ | **6.6552 → 5.8644 (20)** |
| Held-out perplexity | 35.72 | **13.52** |
| Tests | 6/6 | **6/6** |

¹ v1 LoRA is not comparable: the old script left 88% of the model
trainable. v1.0.0 LoRA trains only the adapters, so the loss is higher
but honest.

Raw generations (seed 123, temp 0.8, top_k 40) are in
[`examples/sample_generation.md`](examples/sample_generation.md) and the
full 30-prompt table in
[`examples/benchmark_results.md`](examples/benchmark_results.md). They are
rough — recognisable Bangla words and phrases now appear, but the model
still repeats template phrases and invents word fragments.

## Download demo weights (no training needed)

From the **v1.0.0 release assets** on GitHub:

| File | Size | SHA-256 | What it is |
|---|---|---|---|
| `llm-bd-tiny-demo.pth` | 17.4 MB | `e7601ab6…094b11` | Pretrained tiny demo (200 steps) |
| `llm-bd-sft-demo.pth` | 17.4 MB | `f373a415…50e1bb3` | Tiny demo + 60 steps SFT |

```bash
git clone https://github.com/tunnabri-ops/llm-bd.git
cd llm-bd
pip install -r requirements.txt
# put the two .pth files in out/ as out/tiny_demo.pth and out/sft_demo.pth
python chat.py   # chat in Bangla immediately
```

## How to train the full 64M model

Free Colab T4, ~2 hours — see
[`docs/TRAINING_GUIDE_BN.md`](docs/TRAINING_GUIDE_BN.md) (Bangla) or the
README quickstart:

```bash
python trainer/train_pretrain.py --config 64M --steps 2000 --device cuda \
    --out checkpoints/llm-bd-64M/pretrain.pth
python trainer/train_sft.py --init checkpoints/llm-bd-64M/pretrain.pth \
    --steps 300 --device cuda --out out/sft_64m.pth
```

No 64M benchmark numbers are claimed in this release — they will be
measured after the GPU run and published in the next release notes.

## Known limits

- The tiny demo's Bangla is rough: short, repetitive, sometimes empty or
  nonsensical. It is a pipeline demo, not a usable chatbot.
- The bundled tokenizer (579-token vocab) was trained on the v1 corpus;
  it still handles the bigger corpus (byte-level BPE), but retraining it
  on the expanded data is a planned improvement.
- Template-expanded sentences repeat phrasing — the next data round should
  add more hand-written, non-template text.
- `scripts/convert_to_hf.py` export is best-effort and untested against a
  real transformers load.
