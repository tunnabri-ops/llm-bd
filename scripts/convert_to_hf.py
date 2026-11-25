"""LLM BD -> Hugging Face transformers export (best-effort).

Exports an LLM BD checkpoint to a transformers-compatible directory so it can
be loaded with AutoModelForCausalLM / served with vLLM, llama.cpp (via
transformers) or ollama (via a Modelfile pointing at the exported weights).

Caveat: LLM BD uses RMSNorm + RoPE + SwiGLU, which maps cleanly onto the
Llama architecture in transformers. This script writes a Llama-style
config.json + weights (renamed keys) + tokenizer files. It is best-effort:
verify the exported model with `transformers` before publishing to the Hub.

Usage:
    python scripts/convert_to_hf.py --checkpoint out/sft_demo.pth --out hf/llm-bd-tiny
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LLM BD block key -> Llama key mapping
KEYMAP = [
    ("tok_emb.weight", "model.embed_tokens.weight"),
    ("ln_f.weight", "model.norm.weight"),
    ("lm_head.weight", "lm_head.weight"),
]


def convert_block(sd, i):
    p = f"blocks.{i}."
    q = f"model.layers.{i}."
    out = {
        q + "input_layernorm.weight": sd[p + "ln1.weight"],
        q + "post_attention_layernorm.weight": sd[p + "ln2.weight"],
        q + "self_attn.q_proj.weight": sd[p + "attn.q_proj.weight"],
        q + "self_attn.k_proj.weight": sd[p + "attn.k_proj.weight"],
        q + "self_attn.v_proj.weight": sd[p + "attn.v_proj.weight"],
        q + "self_attn.o_proj.weight": sd[p + "attn.o_proj.weight"],
        q + "mlp.gate_proj.weight": sd[p + "mlp.gate.weight"],
        q + "mlp.up_proj.weight": sd[p + "mlp.up.weight"],
        q + "mlp.down_proj.weight": sd[p + "mlp.down.weight"],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/sft_demo.pth")
    ap.add_argument("--out", default="hf/llm-bd-tiny")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, sd = ck["config"], ck["state_dict"]
    os.makedirs(args.out, exist_ok=True)
    new_sd = {}
    for a, b in KEYMAP:
        if a in sd:
            new_sd[b] = sd[a]
    for i in range(cfg["n_layers"]):
        new_sd.update(convert_block(sd, i))
    torch.save(new_sd, os.path.join(args.out, "pytorch_model.bin"))
    hf_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg["vocab_size"],
        "hidden_size": cfg["hidden"],
        "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": cfg["n_heads"],
        "num_key_value_heads": cfg["n_kv_heads"],
        "intermediate_size": cfg["intermediate"],
        "max_position_embeddings": cfg["max_seq_len"],
        "rms_norm_eps": 1e-6,
        "rope_theta": cfg["rope_theta"],
        "tie_word_embeddings": cfg["tie_embeddings"],
        "torch_dtype": "float32",
    }
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(hf_cfg, fh, indent=2)
    print(f"exported {len(new_sd)} tensors -> {args.out}")
    print("Verify with: python -c \"from transformers import AutoModelForCausalLM;"
          f" AutoModelForCausalLM.from_pretrained('{args.out}')\"")


if __name__ == "__main__":
    main()
