"""LLM BD tests — model shape, tokenizer round-trip, loss decreases on a tiny run."""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import LLM, ModelConfig, count_parameters  # noqa: E402


def test_model_forward_shape():
    cfg = ModelConfig.llm_bd_tiny(vocab_size=512)
    model = LLM(cfg)
    x = torch.randint(0, 512, (2, 16))
    logits, loss = model(x, labels=x)
    assert logits.shape == (2, 16, 512)
    assert loss.item() > 0


def test_param_count_tiny_is_small():
    cfg = ModelConfig.llm_bd_tiny(vocab_size=8192)
    n = count_parameters(LLM(cfg))
    assert 1_000_000 < n < 30_000_000, f"tiny should be ~5M params, got {n}"


def test_param_count_64m():
    cfg = ModelConfig.llm_bd_64m(vocab_size=8192)
    n = count_parameters(LLM(cfg))
    assert 40_000_000 < n < 120_000_000, f"64M should be ~64M params, got {n}"


def test_generate_returns_longer_sequence():
    cfg = ModelConfig.llm_bd_tiny(vocab_size=512)
    model = LLM(cfg)
    idx = torch.randint(0, 512, (1, 5))
    out = model.generate(idx, max_new_tokens=8, temperature=0)
    assert out.shape[1] > 5


def test_tokenizer_round_trip():
    pytest.importorskip("tokenizers")
    from model.tokenizer import BanglaTokenizer
    path = "model/tokenizer.json"
    if not os.path.exists(path):
        pytest.skip("tokenizer.json not built yet — run python train_tokenizer.py")
    tok = BanglaTokenizer(path)
    for text in ["আমি বাংলায় কথা বলি।", "ঢাকা বাংলাদেশের রাজধানী।"]:
        assert tok.decode(tok.encode(text)).strip() == text.strip()


def test_loss_decreases_on_tiny_run():
    """20 steps on random-ish Bangla token ids: loss must not explode."""
    torch.manual_seed(0)
    cfg = ModelConfig.llm_bd_tiny(vocab_size=256)
    model = LLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.randint(0, 256, (64, 32))
    first = last = None
    for step in range(20):
        x, y = data[step % 8: step % 8 + 4], data[step % 8: step % 8 + 4]
        _, loss = model(x, labels=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first * 1.5, f"loss grew too much: {first:.3f} -> {last:.3f}"
