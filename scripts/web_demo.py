"""LLM BD Streamlit Bangla chat WebUI.

Usage:
    streamlit run scripts/web_demo.py
    streamlit run scripts/web_demo.py -- --checkpoint out/sft_demo.pth
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import LLM  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402


def load(checkpoint):
    tok = BanglaTokenizer("model/tokenizer.json")
    model = LLM.load(checkpoint, device="cpu")
    return tok, model


def main():
    import streamlit as st

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/sft_demo.pth")
    args, _ = ap.parse_known_args()
    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        ckpt = "out/tiny_demo.pth"
    st.set_page_config(page_title="LLM BD — বাংলা চ্যাট", page_icon="🇧🇩")
    st.title("🇧🇩 LLM BD — বাংলা ভাষা মডেল")
    st.caption("Train your own Bangla LLM from scratch — 64M model, PyTorch, open source.")
    if not os.path.exists(ckpt):
        st.warning("কোনো ট্রেন করা মডেল পাওয়া যায়নি। আগে চালান: "
                   "`python trainer/train_pretrain.py --tiny --steps 100`")
        st.stop()
    if "tok" not in st.session_state:
        st.session_state.tok, st.session_state.model = load(ckpt)
        st.session_state.history = []
    temp = st.sidebar.slider("Temperature", 0.0, 1.5, 0.8, 0.1)
    max_new = st.sidebar.slider("Max tokens", 16, 256, 64, 16)
    if st.sidebar.button("চ্যাট মুছুন"):
        st.session_state.history = []
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)
    if user := st.chat_input("বাংলায় লিখুন…"):
        st.session_state.history.append(("user", user))
        st.chat_message("user").write(user)
        tok, model = st.session_state.tok, st.session_state.model
        prompt = tok.encode_chat(user)
        ids = torch.tensor([tok.encode(prompt, add_special=False)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=max_new, temperature=temp)
        reply = tok.decode(out[0].tolist())
        # decode() skips special tokens — strip the decoded prompt, not the raw template
        decoded_prompt = tok.decode(ids[0].tolist())
        if reply.startswith(decoded_prompt):
            reply = reply[len(decoded_prompt):]
        reply = reply.strip() or "(খালি উত্তর — আরও ট্রেনিং দরকার)"
        st.session_state.history.append(("assistant", reply))
        st.chat_message("assistant").write(reply)


if __name__ == "__main__":
    main()
