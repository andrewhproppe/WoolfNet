"""
Small GPT-style Transformer for character/word-level language modeling.
Inspired by GPT-2 architecture. Uses SwiGLU activations and RoPE positional embeddings.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from woolfnet.gpt.config import GPTConfig


class GPT(nn.Module):
    """
    GPT-style Transformer model (decoder-only, causal LM)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx):
        x = self.dropout(self.token_embedding(idx))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class Block(nn.Module):
    """
    Single GPT-style decoder block: multi-head self-attention + SwiGLU feedforward
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        hidden = 4 * config.n_embd
        self.mlp_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.mlp_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.mlp_down = nn.Linear(hidden, config.n_embd, bias=False)
        self.mlp_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)
        x = x + self.mlp_drop(self.mlp_down(F.silu(self.mlp_gate(h)) * self.mlp_up(h)))
        return x


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal mask and RoPE positional embeddings.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        # RoPE: precompute per-dim frequencies
        theta = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_theta", theta)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings to x of shape (b, nh, t, head_dim)."""
        t = x.size(2)
        angles = torch.outer(torch.arange(t, device=x.device).float(), self.rope_theta)
        cos = angles.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)  # (1, 1, t, head_dim)
        sin = angles.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # (b, nh, t, hd)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self._apply_rope(q), self._apply_rope(k)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = self.attn_dropout(torch.softmax(att, dim=-1))

        y = att @ v  # (b, nh, t, hd)
        y = self.resid_dropout(self.proj(y.transpose(1, 2).contiguous().view(b, t, c)))
        return y
