"""LLM BD — a small GPT-style decoder language model, written from scratch.

Pure PyTorch, no transformers / no third-party model code. Architecture:
RMSNorm -> causal self-attention with RoPE -> SwiGLU MLP, pre-norm blocks.
Configs:
  * llm-bd-64M : hidden 768, 12 layers, 12 heads  (~64M params)
  * llm-bd-tiny: hidden 256, 4 layers, 4 heads   (~5M params, CPU demo)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    """Plain config object (kept dependency-free on purpose)."""

    def __init__(self, vocab_size=8192, hidden=768, n_layers=12, n_heads=12,
                 n_kv_heads=None, intermediate=None, max_seq_len=1024,
                 dropout=0.0, rope_theta=10000.0, tie_embeddings=True):
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # grouped-query attention when < n_heads
        self.intermediate = intermediate or 4 * hidden
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings
        self.head_dim = hidden // n_heads
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        # to_dict() also stores the derived head_dim — __init__ recomputes it.
        keep = {"vocab_size", "hidden", "n_layers", "n_heads", "n_kv_heads",
                "intermediate", "max_seq_len", "dropout", "rope_theta",
                "tie_embeddings"}
        return cls(**{k: v for k, v in d.items() if k in keep})

    @classmethod
    def llm_bd_64m(cls, vocab_size=8192):
        """llm-bd-64M — the full model (needs a GPU for the 2-hour run).
        hidden 768 / 12 layers / 12 heads, narrow SwiGLU (intermediate 1024)
        so the total really is ~64M params at vocab 8192 (verified in __main__)."""
        return cls(vocab_size=vocab_size, hidden=768, n_layers=12, n_heads=12,
                   intermediate=1024)

    @classmethod
    def llm_bd_tiny(cls, vocab_size=8192):
        """llm-bd-tiny — ~5M params, trains on a laptop CPU in minutes."""
        return cls(vocab_size=vocab_size, hidden=256, n_layers=4, n_heads=4,
                   intermediate=1024, max_seq_len=256)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rope_freqs(head_dim, seq_len, theta, device, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(q, k, cos, sin):
    # q, k: (B, heads, T, head_dim); cos/sin: (T, head_dim/2)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    def rot(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)
    return rot(q), rot(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.hidden, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        if self.n_kv != self.n_heads:  # repeat KV heads for GQA
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        cos, sin = _rope_freqs(self.head_dim, T, self.cfg.rope_theta, x.device, x.dtype)
        q, k = _apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.cfg.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.up = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.hidden, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.hidden)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LLM(nn.Module):
    """LLM BD decoder. Forward returns (logits, loss) — loss when labels given."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.hidden)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, labels=None):
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.8, top_k=40):
        """Greedy when temperature<=0, else top-k sampling. idx: (1, T) long tensor."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            nxt = logits[:, -1, :]
            if temperature <= 0:
                token = nxt.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(nxt / temperature, dim=-1)
                if top_k:
                    v, ix = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs = torch.zeros_like(probs).scatter(-1, ix, v)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, token], dim=1)
            if int(token.item()) == 2:  # <eos>
                break
        return idx

    def save(self, path):
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(ModelConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()


if __name__ == "__main__":
    for name, cfg in [("llm-bd-64M", ModelConfig.llm_bd_64m()),
                      ("llm-bd-tiny", ModelConfig.llm_bd_tiny())]:
        m = LLM(cfg)
        print(f"{name}: {count_parameters(m)/1e6:.1f}M params")
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = m(x, labels=x)
        print(f"  forward ok: logits {tuple(logits.shape)}, loss {loss.item():.3f}")
