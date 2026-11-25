"""LLM BD FastAPI server — OpenAI-compatible /v1/chat/completions.

Usage:
    python scripts/chat_api.py --checkpoint out/sft_demo.pth
    curl -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model": "llm-bd-tiny", "messages": [{"role": "user", "content": "ঢাকা সম্পর্কে বলো"}]}'
"""
import argparse
import os
import sys
import time
import uuid

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import LLM  # noqa: E402
from model.tokenizer import BanglaTokenizer  # noqa: E402

MODEL = None
TOK = None


def create_app(checkpoint):
    global MODEL, TOK
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="LLM BD API", version="0.1.0")
    TOK = BanglaTokenizer("model/tokenizer.json")
    MODEL = LLM.load(checkpoint, device="cpu")

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "llm-bd-tiny"
        messages: list[Message]
        temperature: float = 0.8
        max_tokens: int = 128

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "llm-bd-tiny", "checkpoint": checkpoint}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "llm-bd-tiny", "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        prompt = TOK.encode_chat(user)
        ids = torch.tensor([TOK.encode(prompt, add_special=False)], dtype=torch.long)
        out = MODEL.generate(ids, max_new_tokens=req.max_tokens,
                             temperature=req.temperature)
        reply = TOK.decode(out[0].tolist())
        # decode() skips special tokens — strip the decoded prompt, not the raw template
        decoded_prompt = TOK.decode(ids[0].tolist())
        if reply.startswith(decoded_prompt):
            reply = reply[len(decoded_prompt):]
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply.strip()},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(ids[0]), "completion_tokens": len(out[0]) - len(ids[0]),
                          "total_tokens": len(out[0])}}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/sft_demo.pth")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint} — train first, see README.")
        sys.exit(1)
    import uvicorn
    uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
