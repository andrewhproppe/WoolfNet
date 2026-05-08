import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets):
    """
    Cross entory loss for JAX models
    """
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
    return loss.mean()
