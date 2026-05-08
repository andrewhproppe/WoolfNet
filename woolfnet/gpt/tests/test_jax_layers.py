"""
Test different components of the Jax GPT-2 model
"""

import jax

from woolfnet.gpt.config import GPTConfig
from woolfnet.gpt.model_jax import GPT, Block, CausalSelfAttention

TEST_CONFIG = GPTConfig(
    vocab_size=1000, block_size=16, n_layer=1, n_head=4, n_embd=32, dropout=0.1
)


def test_jax_attention():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 32))  # (batch, sequence, embedding)

    block = CausalSelfAttention(
        n_head=TEST_CONFIG.n_head,
        n_embd=TEST_CONFIG.n_embd,
        dropout=TEST_CONFIG.dropout,
    )

    params = block.init(key, x, deterministic=True)
    y = block.apply(params, x, deterministic=True)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print(
        "Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params))
    )


def test_jax_block():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 32))  # (batch, sequence, embedding)

    block = Block(TEST_CONFIG)

    params = block.init(key, x, deterministic=True)
    y = block.apply(params, x, deterministic=True)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print(
        "Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params))
    )


def test_jax_gpt():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 32))  # (batch, sequence, embedding)

    model = GPT(TEST_CONFIG)

    params = model.init(key, x)
    y = model.apply(params, x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print(
        "Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params))
    )
