# LLM BD — ট্রেনিং গাইড (বাংলা)

নিজের বাংলা LLM শূন্য থেকে ট্রেন করার ধাপে ধাপে নির্দেশনা — ফ্রি Colab T4 GPU
অথবা নিজের ল্যাপটপের CPU, দুটোতেই চলবে।

## ১. যা যা লাগবে

- Python ৩.১০ বা তার বেশি
- ইন্টারনেট (প্রথমবার প্যাকেজ ডাউনলোডের জন্য)
- Colab ব্যবহার করলে: একটি ফ্রি Google অ্যাকাউন্ট (GPU-এর জন্য কোনো টাকা লাগে না)

## ২. সেটআপ

```bash
git clone https://github.com/tunnabri-ops/llm-bd.git
cd llm-bd
pip install -r requirements.txt
```

## ৩. ডেটাসেট ও টোকেনাইজার

```bash
python dataset/build_dataset.py   # ১০১৬টি প্রিট্রেন বাক্য + ৩২৫টি SFT জোড়া তৈরি হয়
python train_tokenizer.py         # বাংলা BPE টোকেনাইজার তৈরি হয়
```

নিজের বাংলা লেখা যোগ করতে চাইলে:

```bash
python dataset/build_dataset.py --extra-text আমার_লেখা.txt
```

`আমার_লেখা.txt`-এ প্রতি লাইনে একটি করে বাক্য লিখুন।

## ৪. ছোট ডেমো (CPU, কয়েক মিনিট)

```bash
python trainer/train_pretrain.py --tiny --steps 200   # লস কমতে দেখুন
python trainer/train_sft.py --init out/tiny_demo.pth --steps 60
python chat.py
```

> টিপস: `--steps` বাড়ালে মডেল ভালো হয়, তবে CPU-তে সময় বেশি লাগে।
> ২০০ স্টেপে প্রিট্রেনিং লস শুরুতে ~৬.৪ থেকে শেষে ~২.৫-এর কাছাকাছি আসে।

## ৫. পুরো মডেল (Colab T4 GPU, প্রায় ২ ঘণ্টা)

Colab-এ `train.ipynb` খুলুন (README-র Colab ব্যাজে ক্লিক করুন), তারপর
Runtime → Change runtime type → GPU (T4) নির্বাচন করুন। তারপর Run all।

অথবা নিজের GPU থাকলে টার্মিনালে:

```bash
python trainer/train_pretrain.py --config 64M --steps 2000 --device cuda \
    --out checkpoints/llm-bd-64M/pretrain.pth
python trainer/train_sft.py --init checkpoints/llm-bd-64M/pretrain.pth \
    --steps 300 --device cuda --out out/sft_64m.pth
python chat.py --checkpoint out/sft_64m.pth
```

## ৬. LoRA (অল্প প্যারামিটারে ফাইন-টিউন)

```bash
python trainer/train_lora.py --init out/tiny_demo.pth --steps 20 --rank 8
```

LoRA শুধু ছোট অ্যাডাপ্টার ট্রেন করে — পুরো মডেল জমে থাকে, তাই কম মেমরি লাগে।

## ৭. মূল্যায়ন

```bash
python eval_llm.py --checkpoint out/tiny_demo.pth --perplexity
python scripts/benchmark.py --checkpoint out/tiny_demo.pth
```

প্রথমটি হেল্ড-আউট বাংলা লেখায় পারপ্লেক্সিটি দেয়, দ্বিতীয়টি ৩০টি ডেমো
প্রম্পটের আসল আউটপুট `examples/benchmark_results.md`-এ লিখে দেয়।

## ৮. চ্যাট, API, ওয়েব UI

```bash
python chat.py                                  # টার্মিনালে বাংলায় চ্যাট
streamlit run scripts/web_demo.py               # ব্রাউজারে চ্যাট
python scripts/chat_api.py --checkpoint out/sft_demo.pth   # OpenAI-সামঞ্জস্য API
```

## ৯. ডেমো ওয়েট ডাউনলোড করে ট্রেনিং ছাড়াই চ্যাট

ট্রেনিং না করে এখনই চেষ্টা করতে চাইলে GitHub Releases থেকে
`llm-bd-tiny-demo.pth` ও `llm-bd-sft-demo.pth` ডাউনলোড করুন, `out/` ফোল্ডারে
রাখুন, তারপর `python chat.py` চালান। বিস্তারিত:
[RELEASE_NOTES_v1.0.0.md](RELEASE_NOTES_v1.0.0.md)।

## ১০. সমস্যা হলে

| সমস্যা | সমাধান |
|---|---|
| `torch` ইনস্টল হচ্ছে না | `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU) |
| Colab-এ GPU পাচ্ছি না | Runtime → Change runtime type → T4 GPU, তারপর আবার রান করুন |
| মডেলের উত্তর অগোছালো | ছোট ডেমোতে এটাই স্বাভাবিক — বেশি স্টেপ বা 64M মডেল ট্রেন করুন, আর বেশি বাংলা ডেটা যোগ করুন |
| মেমরি শেষ হয়ে যাচ্ছে | `--batch 4` ও `--seq-len 128` ব্যবহার করুন |
