"""
GPT2 model in JAX using Flax NNX.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class CausalSelfAttention(nnx.Module):
    def __init__(self, n_head: int, n_embd: int, dropout: float, *, rngs: nnx.Rngs):
        self.n_head = n_head
        self.n_embd = n_embd

        self.qkv_proj = nnx.Linear(n_embd, 3 * n_embd, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.dropout_layer = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        B, T, C = x.shape
        head_dim = C // self.n_head

        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        att = jnp.where(mask, att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1)
        att = self.dropout_layer(att, deterministic=not training)

        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.out_proj(y)
        y = self.dropout_layer(y, deterministic=not training)
        return y


class Block(nnx.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=dropout, rngs=rngs)
        self.fc1 = nnx.Linear(n_embd, 4 * n_embd, rngs=rngs)
        self.fc2 = nnx.Linear(4 * n_embd, n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        x = x + self.attn(self.ln1(x), training)
        y = self.fc1(self.ln2(x))
        y = jax.nn.gelu(y)
        y = self.fc2(y)
        y = self.dropout(y, deterministic=not training)
        return x + y


class GPT(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.block_size = block_size

        self.token_emb = nnx.Embed(vocab_size, n_embd, rngs=rngs)
        self.pos_emb = nnx.Embed(block_size, n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        self.blocks = nnx.List([Block(n_embd, n_head, dropout, rngs=rngs) for _ in range(n_layer)])
        self.ln_f = nnx.LayerNorm(n_embd, rngs=rngs)
        self.head = nnx.Linear(n_embd, vocab_size, rngs=rngs)

    def __call__(self, idx: jax.Array, training: bool = False) -> jax.Array:
        b, t = idx.shape

        token_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(jnp.arange(t))
        x = token_emb + pos_emb
        x = self.dropout(x, deterministic=not training)

        for block in self.blocks:
            x = block(x, training)

        x = self.ln_f(x)
        return self.head(x)
