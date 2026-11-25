# LLM BD — Train Your Own Bangla LLM From Scratch in 2 Hours 🇧🇩

> **One line: a 64M Bangla language model you train yourself, from zero, in one evening — no API key, no fine-tuning someone else's model, no black box.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnabri-ops/llm-bd/blob/main/train.ipynb)
[![CI](https://github.com/tunnabri-ops/llm-bd/actions/workflows/ci.yml/badge.svg)](https://github.com/tunnabri-ops/llm-bd/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**[বাংলা README](README-BN.md) | [Colab notebook](train.ipynb) | [Demo prompts](examples/demo_prompts_bn.md)**

Bangla is spoken by 230+ million people — yet almost every open LLM tutorial
trains in English or Chinese. **LLM BD** fixes that: a complete, tiny-LLM-from-scratch
project where the language is Bangla, the code is pure PyTorch, and every step
is small enough to read in one sitting.

---

## Who is this for?

| You are… | You get… |
|---|---|
| Learning how LLMs actually work | Every line of the model, tokenizer and trainer — readable, no magic |
| A Bangla speaker who codes | The first from-scratch LLM project in your own language |
| A student with one GPU (or a free Colab T4) | A real 64M model trained in ~2 hours |
| A teacher / club lead | A full 2-hour workshop plan (see the roadmap below) |

## Why this goes viral

1. **First of its kind for Bangla** — 230M speakers, almost zero from-scratch LLM projects in the language.
2. **Tiny but real** — 64M parameters fits on a free Colab GPU; the `llm-bd-tiny` demo (~5M) trains on a laptop CPU in minutes.
3. **No black boxes** — RMSNorm, RoPE, SwiGLU, grouped-query attention, pretrain → SFT → LoRA, all written from scratch in PyTorch. You can read the whole model in one file (`model/model.py`).
4. **It talks back in Bangla** — CLI chat, Streamlit WebUI, and an OpenAI-compatible API server, all included.

## Quickstart

```bash
git clone https://github.com/tunnabri-ops/llm-bd.git
cd llm-bd
pip install -r requirements.txt

python dataset/build_dataset.py   # 207 pretrain sentences + 132 SFT pairs, Bangla
python train_tokenizer.py         # Bangla BPE tokenizer (target vocab 8192)
python trainer/train_pretrain.py --tiny --steps 100   # CPU demo, minutes
python chat.py                    # chat in Bangla
```

Full 64M run (needs a GPU — free Colab T4 works):

```bash
python trainer/train_pretrain.py --config 64M --steps 2000 --device cuda \
    --out checkpoints/llm-bd-64M/pretrain.pth
python trainer/train_sft.py --init checkpoints/llm-bd-64M/pretrain.pth \
    --steps 300 --device cuda --out out/sft_64m.pth
python chat.py --checkpoint out/sft_64m.pth
```

Or run it all in the browser: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnabri-ops/llm-bd/blob/main/train.ipynb) — open `train.ipynb`, Runtime → Run all.

## Architecture

```
tokens ──► Embedding ──► ┌──────────────────────────────┐ ──► RMSNorm ──► LM head ──► logits
                         │  ×12 blocks (llm-bd-64M)    │
                         │  RMSNorm → RoPE attention    │
                         │  (grouped-query, causal)     │
                         │  RMSNorm → SwiGLU MLP        │
                         └──────────────────────────────┘
```

- **Attention:** causal self-attention with Rotary Position Embeddings (RoPE), optional grouped-query attention (`n_kv_heads < n_heads`).
- **Norm / MLP:** RMSNorm (pre-norm), SwiGLU feed-forward.
- **Training:** AdamW, cosine LR schedule with warmup, checkpoint save/resume, gradient clipping.
- **Tokenizer:** byte-level BPE trained on Bangla text; special tokens `<pad> <bos> <eos> <im_start> <im_end> <think> <tool_call>`. The bundled sample corpus yields a 579-token vocab; with a larger corpus it grows toward the 8192 target.

## Models

| Model | Params | Layers | Hidden | Heads | Sequence | For |
|---|---|---|---|---|---|---|
| **llm-bd-64M** | ~64M | 12 | 768 | 12 | 1024 | The real model — GPU, ~2 h |
| **llm-bd-tiny** | ~5M | 4 | 256 | 4 | 256 | CPU demo, minutes |

Check the counts yourself: `python model/model.py` prints both.

## The 2-hour roadmap

| Time | What you do |
|---|---|
| 0:00–0:15 | Setup, build the dataset, train the tokenizer |
| 0:15–0:30 | Train `llm-bd-tiny` on CPU — watch the loss fall |
| 0:30–1:30 | Pretrain `llm-bd-64M` on GPU (the long step — grab tea) |
| 1:30–1:50 | SFT: teach it to answer questions in Bangla |
| 1:50–2:00 | Chat, evaluate, share your model |

## What it actually does (measured, not promised)

The tiny demo was really trained on CPU in this repo — loss, steps and raw
generations are in [`examples/sample_generation.md`](examples/sample_generation.md):

- **Pretrain (`llm-bd-tiny`, 100 steps, CPU, ~2 min): loss 6.3745 → 3.0293**
- **Held-out perplexity: 35.72** (`dataset/pretrain_val.jsonl`)
- **SFT (30 steps): loss 5.5351 → 2.6178 · LoRA (10 steps): 5.5351 → 3.0786**
- **Checkpoint: `out/tiny_demo.pth`, 17.4 MB** (weights only)
- **Tests: 6/6 pass** (`pytest tests/ -q`)

It is a ~4.3M model on a 207-sentence corpus: its Bangla is rough and it
hallucinates. That is the honest baseline — the point of the project is that
*you* can see the loss fall and then scale it up.

Benchmark method (run it yourself):

```bash
python eval_llm.py --checkpoint out/tiny_demo.pth --perplexity   # held-out Bangla PPL
python eval_llm.py --checkpoint out/tiny_demo.pth --prompt "ঢাকা বাংলাদেশের"
pytest tests/ -q
```

Full 64M benchmarks (held-out perplexity + the 20 demo prompts in
`examples/demo_prompts_bn.md`) are measured after the GPU run and will be
published in the release notes — no numbers are claimed before they are run.

## Chat, API, WebUI

```bash
python chat.py                                  # interactive Bangla CLI chat
streamlit run scripts/web_demo.py               # Bangla chat in the browser
python scripts/chat_api.py --checkpoint out/sft_demo.pth   # OpenAI-compatible API
# POST /v1/chat/completions  {"model": "llm-bd-tiny", "messages": [{"role": "user", "content": "ঢাকা সম্পর্কে বলো"}]}
```

Also included: `trainer/train_lora.py` (LoRA from scratch, no peft),
`scripts/convert_to_hf.py` (export to transformers/Llama format, best-effort),
`scripts/download_data.py` (fetch a bigger Bangla corpus to scale up).

## FAQ

**Do I need a GPU?** For `llm-bd-tiny`, no — any laptop CPU works. For the
full 64M in 2 hours, yes — a free Colab T4 is enough.

**How good is the tiny model?** Honestly: rough. It is a demo that proves the
pipeline works end-to-end. Fluency comes from the 64M run + more Bangla data.

**Can I add my own Bangla text?** Yes — add a text file and run
`python dataset/build_dataset.py --extra-text my_bangla.txt`, then retrain
the tokenizer. Contributions of Bangla text are the most welcome PRs.

**Is this a fine-tune of another model?** No. Weights start random; the model
learns Bangla from the corpus in this repo. That is the whole point.

**License?** Apache 2.0 — free for personal and commercial use.

## Contributing

Bangla data, bug fixes, better evals — all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

**Topics:** `bangla` `llm` `language-model` `pytorch` `train-from-scratch` `ai` `bengali` `nlp` `small-language-model` `llm-training`
