# LLM BD — Sample Generation (real tiny-demo run, v1.0.0)

Model: `llm-bd-tiny` (~4.3M params at the bundled tokenizer vocab of 579),
trained on CPU for **200 steps** on `dataset/pretrain_bangla.jsonl`
(1,016 sentences, 54,332 tokens — the v1.1 expanded corpus).

- Loss: **6.4114 → 2.2212** (start → end, 200 steps, AdamW + cosine LR, batch 8, seq 128)
- SFT (60 steps on 325 pairs): **6.6552 → 1.4696**
- LoRA (20 steps, rank 8, 0.75% of params trainable): **6.6552 → 5.8644**
- Held-out perplexity (`dataset/pretrain_val.jsonl`, 20 lines, none in train): **13.52**
- Checkpoint: `out/tiny_demo.pth` (weights only, **17.4 MB**)
- Generation settings: `temperature=0.8`, `top_k=40`, `max_new_tokens=64`, `torch.manual_seed(123)`

These outputs are **real and unedited** — the tiny model is still undertrained,
so its Bangla is rough and repetitive, but you can see real Bangla words and
template phrases ("বাংলাদেশের", "কারণ এটি") starting to emerge. That is the
honest state: the pipeline works end-to-end, and fluency is what the full
`llm-bd-64M` GPU run + more Bangla data are for.

Full 30-prompt benchmark: [`benchmark_results.md`](benchmark_results.md).

## Sample 1

**Prompt:** `ঢাকা বাংলাদেশের`

**Output:**

```
ঢাকা বাংলাদেশেরবাংলাদেশের কাহের প্ণ।
```

## Sample 2

**Prompt:** `বাংলা ভাষা`

**Output:**

```
বাংলা ভাষাবাংলাদেশের কাহের প্ণাওয়া মারিয়া কারণ এটি ভূগেলে বাংলাদেশের তথ্বপূগোর একটিচয়েই সবাপুত
```

## Sample 3

**Prompt:** `সুন্দরবন`

**Output:**

```
সুন্দরবনটকগারিকে জালিলে পাহাওয়া মজিহাসিক জানুড়পুরাখতে পাওয়া সহজ, কানে মান্রিগবনিহাত
```

## Sample 4

**Prompt:** `কৃত্রিম বুদ্ধিমত্তা`

**Output:**

```
কৃত্রিম বুদ্ধিমত্তাবাংলাদেশের কাহের প্ণ।
```

## Sample 5

**Prompt:** `ক্রিকেট খেলা`

**Output:**

```
ক্রিকেট খেলাবাংলাদেশের কাহস্রিয়া সুবদ্রাস কারণ এটি ভূর জন্য বিলাদেশের তথ্বপূগোর একটিচয়েই মনে আসে।
```

---

## v1.0 baseline (for comparison)

The first tiny run, before the dataset expansion — 100 steps on 207
sentences (10,790 tokens), 10-line val set that overlapped train:

- Loss: **6.3745 → 3.0293** (100 steps, batch 8, seq 128)
- Held-out perplexity: **35.72**
- SFT (30 steps): 5.5351 → 2.6178 · LoRA (10 steps): 5.5351 → 3.0786
  (note: the v1 LoRA number is not comparable — the v1 LoRA script left
  88% of the model trainable; v1.0.0 fixes it so only the 0.75% adapter
  weights train)

### v1.0 Sample 1

**Prompt:** `ঢাকা বাংলাদেশের`

**Output:**

```
ঢাকা বাংলাদেশেরবুশেরিপর্র্বপ় দরা বিত্কেরন্কেনাইইই একে জানে।
```

### v1.0 Sample 2

**Prompt:** `বাংলা ভাষা`

**Output:**

```
বাংলা ভাষারি সীতেশি্রথমের় সিি জানের়র প্লালালাদেরত্বেরেরেশেশ্রারাদেরায্নদে্ষিশ
```

### v1.0 Sample 3

**Prompt:** `সুন্দরবন`

**Output:**

```
সুন্দরবনজ্রীরি।
```

**What improved in v1.0.0:** pretrain loss 3.03 → 2.22, perplexity 35.72 →
13.52, and generations now contain recognisable Bangla words and phrases
instead of mostly broken characters — with 5× more data and 2× the steps.
Still rough: the model repeats the template phrase "বাংলাদেশের" and
invents word fragments. The 64M GPU run is the next step.
