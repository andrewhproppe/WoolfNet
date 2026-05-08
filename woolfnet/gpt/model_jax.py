"""
GPT2 model in JAX using Flax Linen.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn import initializers

from woolfnet.gpt.config import GPTConfig

JAX_ACTIVATION_REGISTRY = {
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
}


# --- using Flax linen ---


def DenseMetal(features, use_bias=True, kernel_init=initializers.normal(stddev=0.02)):
    """
    Wrapper for Dense layers to use Apple silicon. Avoids using erf in the kernel
    initializer.
    """
    return nn.Dense(features, use_bias=use_bias, kernel_init=kernel_init)


class MLPBlock(nn.Module):
    """
    Block for multilayer perceptron that combines dense, activation, and dropout layers.
    """

    dim: int
    activation: str
    dropout_rate: float

    @nn.compact
    def __call__(self, x):
        net = nn.Sequential(
            [
                DenseMetal(self.dim),
                JAX_ACTIVATION_REGISTRY[self.activation],
                nn.Dropout(rate=self.dropout_rate, deterministic=True),
            ]
        )
        x = net(x)
        return x


class MLPStack(nn.Module):
    """
    Multilayer perceptron.
    """

    hidden_dim: int
    output_dim: int
    num_layers: int
    activation: str
    dropout_rate: float

    @nn.compact
    def __call__(self, x):
        net = [
            MLPBlock(
                dim=self.hidden_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers - 1)
        ]
        net.append(
            MLPBlock(
                dim=self.output_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )
        )
        x = nn.Sequential(net)(x)
        return x


class CausalSelfAttention(nn.Module):
    n_head: int
    n_embd: int
    dropout: float

    @nn.compact
    def __call__(self, x, training: bool):
        B, T, C = x.shape
        head_dim = C // self.n_head

        qkv = DenseMetal(3 * C, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        att = jnp.where(mask, att, -jnp.inf)

        att = jax.nn.softmax(att, axis=-1)
        att = nn.Dropout(rate=self.dropout)(att, deterministic=not training)

        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = DenseMetal(C, use_bias=False)(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not training)
        return y


class Block(nn.Module):
    """
    Single GPT-style decoder block: multi-head self-attention + feedforward MLP.
    """

    config: GPTConfig

    @nn.compact
    def __call__(self, x, training: bool):
        x_norm = nn.LayerNorm()(x)
        attn_out = CausalSelfAttention(
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            dropout=self.config.dropout,
        )(x_norm, training)
        x = x + attn_out

        x_norm = nn.LayerNorm()(x)
        y = DenseMetal(4 * self.config.n_embd)(x_norm)
        y = jax.nn.gelu(y)
        y = DenseMetal(self.config.n_embd)(y)
        y = nn.Dropout(rate=self.config.dropout)(y, deterministic=not training)

        x = x + y
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, training: bool):
        b, t = idx.shape
        token_emb = nn.Embed(self.config.vocab_size, self.config.n_embd)(idx)
        pos_emb = nn.Embed(self.config.block_size, self.config.n_embd)(jnp.arange(t))
        x = token_emb + pos_emb
        x = nn.Dropout(self.config.dropout, deterministic=not training)(x)

        for _ in range(self.config.n_layer):
            x = Block(self.config)(x=x, training=training)

        x = nn.LayerNorm()(x)
        logits = DenseMetal(self.config.vocab_size)(x)
        return logits
