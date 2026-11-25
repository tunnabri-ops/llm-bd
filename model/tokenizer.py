"""LLM BD tokenizer — Bangla BPE, built with the `tokenizers` library.

Vocab 8192, byte-level BPE (handles Bangla script + Latin + digits natively).
Special tokens: <pad>=0 <bos>=1 <eos>=2 <unk>=3 <im_start>=4 <im_end>=5
                <think>=6 <tool_call>=7
"""
import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<im_start>", "<im_end>",
                  "<think>", "<tool_call>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3


def train_bpe_tokenizer(text_files, vocab_size=8192, save_path="model/tokenizer.json"):
    """Train a byte-level BPE tokenizer on Bangla text files (one sentence per line
    or .jsonl with a 'text' field). Returns the Tokenizer."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
        min_frequency=2, show_progress=True)
    tok.train(files=text_files, trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 1), ("<eos>", 2)])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tok.save(save_path)
    return tok


class BanglaTokenizer:
    """Thin wrapper: encode(text)->ids, decode(ids)->text, chat template helpers."""

    def __init__(self, path="model/tokenizer.json"):
        self.tok = Tokenizer.from_file(path)
        self.vocab_size = self.tok.get_vocab_size()
        self.pad_id, self.bos_id, self.eos_id = PAD, BOS, EOS

    def encode(self, text, add_special=True):
        ids = self.tok.encode(text).ids
        if not add_special:
            # strip the <bos>/<eos> added by the post-processor
            ids = [i for i in ids if i not in (BOS, EOS)]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (BOS, EOS, PAD)]
        return self.tok.decode(ids)

    def encode_chat(self, instruction, inp="", max_len=512):
        """<im_start>user ... <im_end> <im_start>assistant ... — prompt part is
        returned separately so SFT can mask it (see trainer/train_sft.py)."""
        user = f"<im_start>user\n{instruction}"
        if inp:
            user += f"\n{inp}"
        user += "\n<im_end>\n<im_start>assistant\n"
        return user

    @classmethod
    def train_and_save(cls, text_files, vocab_size=8192, save_path="model/tokenizer.json"):
        train_bpe_tokenizer(text_files, vocab_size, save_path)
        return cls(save_path)


def round_trip_check(path="model/tokenizer.json", texts=None):
    tok = BanglaTokenizer(path)
    texts = texts or ["আমি বাংলায় কথা বলি।", "ঢাকা বাংলাদেশের রাজধানী।",
                      "Hello, how are you? আমি ভালো আছি।"]
    ok = True
    for t in texts:
        back = tok.decode(tok.encode(t))
        match = back.strip() == t.strip()
        ok &= match
        print(f"  {'OK ' if match else 'DIFF'} {t!r} -> {back!r}")
    return ok


if __name__ == "__main__":
    import sys
    print("round-trip check:", round_trip_check(sys.argv[1] if len(sys.argv) > 1 else "model/tokenizer.json"))
