# LLM BD — ২ ঘণ্টায় নিজের বাংলা LLM বানান

> **নিজের বাংলা ভাষা মডেল, একদম শূন্য থেকে। কোনো API কী নেই, অন্যের মডেল ফাইন-টিউন নয় — পুরো মডেলটাই আপনি নিজে বানাবেন।**

[English README](README.md) | [Colab-এ খুলুন](https://colab.research.google.com/github/tunnabri-ops/llm-bd/blob/main/train.ipynb)

## কী আছে এখানে?

**LLM BD** একটি ছোট বাংলা ভাষা মডেল — **llm-bd-64M** (৬৪ মিলিয়ন প্যারামিটার), যা আপনি নিজের কম্পিউটারে বা ফ্রি Colab GPU-তে প্রায় ২ ঘণ্টায় শূন্য থেকে ট্রেন করতে পারবেন। সাথে আছে **llm-bd-tiny** (~৫M প্যারামিটার) — সাধারণ ল্যাপটপের CPU-তেই কয়েক মিনিটে চলে, শেখার জন্য আদর্শ।

সব কোড PyTorch-এ শূন্য থেকে লেখা — RMSNorm, RoPE, SwiGLU, গ্রুপড-কোয়েরি অ্যাটেনশন, প্রিট্রেনিং, SFT, LoRA, সবকিছু। কোনো বড় ফ্রেমওয়ার্কের আড়ালে কিছু লুকানো নেই।

## দ্রুত শুরু

```bash
git clone https://github.com/tunnabri-ops/llm-bd.git
cd llm-bd
pip install -r requirements.txt

python dataset/build_dataset.py   # ২০৭টি প্রিট্রেন বাক্য + ১৩২টি SFT প্রশ্নোত্তর
python train_tokenizer.py         # বাংলা BPE টোকেনাইজার
python trainer/train_pretrain.py --tiny --steps 100   # CPU ডেমো
python chat.py                    # বাংলায় চ্যাট করুন
```

GPU থাকলে পুরো মডেল:

```bash
python trainer/train_pretrain.py --config 64M --steps 2000 --device cuda
python trainer/train_sft.py --init checkpoints/llm-bd-64M/pretrain.pth --steps 300 --device cuda
```

## মডেল

| মডেল | প্যারামিটার | লেয়ার | হিডেন | উদ্দেশ্য |
|---|---|---|---|---|
| llm-bd-64M | ~৬৪M | ১২ | ৭৬৮ | আসল মডেল — GPU, ২ ঘণ্টা |
| llm-bd-tiny | ~৫M | ৪ | ২৫৬ | CPU ডেমো, কয়েক মিনিট |

## কী শিখবেন?

১. **টোকেনাইজার** — বাংলা BPE কীভাবে কাজ করে
২. **প্রিট্রেনিং** — পরের শব্দ অনুমান করে ভাষা শেখা
৩. **SFT** — নির্দেশনা অনুযায়ী উত্তর দিতে শেখানো
৪. **LoRA** — অল্প প্যারামিটার দিয়ে ফাইন-টিউন
৫. **ডেপ্লয়** — FastAPI সার্ভার + Streamlit চ্যাট UI

## ২ ঘণ্টার রোডম্যাপ

| সময় | কাজ |
|---|---|
| ০:০০–০:১৫ | সেটআপ, ডেটাসেট, টোকেনাইজার |
| ০:১৫–১:৩০ | প্রিট্রেনিং (llm-bd-64M) |
| ১:৩০–১:৫০ | SFT |
| ১:৫০–২:০০ | চ্যাট, ইভালুয়েশন, শেয়ার |

## ডেমো আউটপুট

আসল ট্রেনিং রানের আউটপুট দেখুন: [examples/sample_generation.md](examples/sample_generation.md)

## লাইসেন্স

Apache 2.0 — সম্পূর্ণ ফ্রি, বাণিজ্যিক ব্যবহারসহ।
