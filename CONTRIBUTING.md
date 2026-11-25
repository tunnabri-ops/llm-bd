# Contributing to LLM BD

Thanks for helping make Bangla AI open to everyone. 🇧🇩

## How to contribute

1. **Fork** the repo and create a branch: `git checkout -b my-change`
2. **Bangla data first** — the best contribution is more Bangla text:
   add sentences to `dataset/build_dataset.py` (pretrain seeds) or Q&A pairs
   (SFT seeds), then run `python dataset/build_dataset.py` and check the counts.
3. **Code** — keep it dependency-light and from-scratch where the model is
   concerned. No `transformers` / `peft` in the core training path.
4. **Test** — `python -m compileall -q model trainer scripts tests *.py`
   and `pytest tests/ -q` must pass. If you touched training, run the tiny
   smoke run: `python trainer/train_pretrain.py --tiny --steps 60`.
5. **Open a PR** with a short description in English or Bangla, and the
   command outputs that prove it works (loss before/after, sample output).

## Style

- Python: readable > clever. Type hints where they help, comments in
  English or Bangla.
- Bangla text: standard Bangla, no Romanised Bangla in datasets.
- No copied code from other LLM-from-scratch projects — write it yourself
  or link the idea, don't paste the implementation.

## Reporting issues

Open an issue with: what you ran, the exact command, the error output,
and your Python / torch versions. Screenshots of Bangla output help a lot.
