# LLM BD — Launch Kit (v1.0.0)

Ready-to-post texts for the v1.0.0 release. Every claim below is measured in
this repo — see [`examples/sample_generation.md`](examples/sample_generation.md)
and [`examples/benchmark_results.md`](examples/benchmark_results.md). No
unmeasured numbers, no hype.

Repo: **https://github.com/tunnabri-ops/llm-bd**

---

## 1. X / Twitter (short)

> I trained a Bangla LLM from scratch — no fine-tuning someone else's model.
>
> LLM BD: 64M params, pure PyTorch, Bangla data, pretrain → SFT → LoRA, all
> readable in one sitting. The tiny demo trains on a laptop CPU in minutes.
>
> Repo + demo weights: https://github.com/tunnabri-ops/llm-bd
>
> Bangla is spoken by 230M+ people. It deserves its own from-scratch LLM.
> 🇧🇩

## 2. LinkedIn (medium)

> **We trained a language model in Bangla — from zero.**
>
> LLM BD is a from-scratch LLM project where the language is Bangla: a 64M
> model in pure PyTorch (RMSNorm, RoPE, SwiGLU, grouped-query attention),
> with a Bangla BPE tokenizer, 1,000+ hand-written pretrain sentences and
> 300+ SFT pairs, and a full pretrain → SFT → LoRA pipeline you can read
> in one sitting.
>
> What makes it different: nothing is a black box. Weights start random.
> The tiny demo (~5M params) trains on a laptop CPU in minutes, and the
> demo weights are downloadable from the v1.0.0 release — you can chat in
> Bangla immediately, no training needed.
>
> Honest note: the tiny demo's Bangla is rough. That's the point — the
> repo shows the real loss curve, real perplexity, and real raw outputs,
> so you can see exactly where it is and scale it up yourself on a free
> Colab T4.
>
> Repo: https://github.com/tunnabri-ops/llm-bd
>
> If you speak Bangla, or you teach ML, the most welcome contribution is
> more Bangla text. 🇧🇩

## 3. Facebook / Bangla

> 🇧🇩 **নিজের বাংলা ভাষা মডেল — একদম শূন্য থেকে!**
>
> LLM BD: একটি বাংলা LLM, যা অন্যের মডেল ফাইন-টিউন করে নয় — একদম শুরু
> থেকে, শুধু PyTorch দিয়ে বানানো। ৬৪M প্যারামিটার, বাংলা টোকেনাইজার,
> ১০০০+ বাংলা বাক্য ও ৩০০+ প্রশ্নোত্তর দিয়ে ট্রেন করা।
>
> ছোট ডেমো মডেলটি ল্যাপটপের CPU-তেই কয়েক মিনিটে ট্রেন হয় — আর v1.0.0
> রিলিজ থেকে ডেমো ওয়েট ডাউনলোড করে ট্রেনিং ছাড়াই এখনই বাংলায় চ্যাট
> করা যায়।
>
> সত্যি কথা: ছোট মডেলটির বাংলা এখনো অগোছালো। কিন্তু রেপোতে আসল লস,
> আসল পারপ্লেক্সিটি আর আসল আউটপুট দেওয়া আছে — তাই আপনি নিজেই দেখতে
> পারবেন, আর ফ্রি Colab GPU-তে বড় মডেল ট্রেন করে ভালো করতে পারবেন।
>
> রেপো: https://github.com/tunnabri-ops/llm-bd
>
> বাংলা ভাষায় আরও লেখা দিলে মডেলটি আরও ভালো হবে — আপনার লেখা পাঠান!

---

## Posting checklist

1. Post from the repo's release page: link the **v1.0.0 release**, not just
   the repo root, so people find the downloadable demo weights.
2. Pin the repo link in the first comment if the platform hides links.
3. Reply to the first questions with the quickstart from the README —
   `python chat.py` after downloading the demo weights.
4. Do not quote any benchmark number that is not in
   `examples/benchmark_results.md` or `examples/sample_generation.md`.
