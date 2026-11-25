# LLM BD — Sample Generation (real tiny-demo run)
Model: `llm-bd-tiny` (~4.3M params at the sample-corpus vocab of 579), trained on CPU for **100 steps** on `dataset/pretrain_bangla.jsonl` (207 sentences, 10,790 tokens).
- Loss: **6.3745 → 3.0293** (start → end, 100 steps, AdamW + cosine LR, batch 8, seq 128)
- Held-out perplexity (`dataset/pretrain_val.jsonl`, 10 sentences): **35.72**
- Checkpoint: `out/tiny_demo.pth` (weights only, **17.4 MB**)
- Generation settings: `temperature=0.8`, `top_k=40`, `max_new_tokens=64`, `torch.manual_seed(123)`

These outputs are **real and unedited** — the tiny model is undertrained on a 207-sentence corpus, so its Bangla is rough, repetitive and sometimes stops early (it has learned `<eos>` too well). That is the honest baseline: the pipeline works end-to-end, and fluency is what the full `llm-bd-64M` GPU run + more Bangla data are for.

## Sample 1

**Prompt:** `ঢাকা বাংলাদেশের`

**Output:**

```
ঢাকা বাংলাদেশেরবুশেরিপর্র্বপ় দরা বিত্কেরন্কেনাইইই একে জানে।
```

## Sample 2

**Prompt:** `বাংলা ভাষা`

**Output:**

```
বাংলা ভাষারি সীতেশি্রথমের় সিি জানের়র প্লালালাদেরত্বেরেরেশেশ্রারাদেরায্নদে্ষিশ
```

## Sample 3

**Prompt:** `সুন্দরবন`

**Output:**

```
সুন্দরবনজ্রীরি।
```
