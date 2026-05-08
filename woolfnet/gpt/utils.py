"""
Utility functions for GPT model
"""

import logging
from pathlib import Path

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import mlflow
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def warmup_optimizer(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    base_lr: float,
    warmup_start_lr: float,
):
    """
    Update the optimizer lr based on current epoch, number of warmup epochs, and the
    learning rates.
    """
    warmup_lr = (
        warmup_start_lr + (base_lr - warmup_start_lr) * (epoch + 1) / warmup_epochs
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = warmup_lr

    return optimizer


def generate_text(
    model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device="cpu"
):
    """
    Generate text from a GPT-style model given a prompt.

    model: trained GPT model
    tokenizer: your ByteLevelBPETokenizer
    prompt: string
    max_new_tokens: how many new tokens to generate
    temperature: randomness control; 1.0 is default
    """
    model.eval()

    # Encode prompt
    input_ids = torch.tensor(
        tokenizer.encode(prompt).ids, dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, seq_len)

    generated = input_ids

    for _ in range(max_new_tokens):
        # Truncate if input longer than block size
        x_cond = generated[:, -model.config.block_size :]

        with torch.no_grad():
            logits = model(x_cond)

        # Take logits of last token
        logits_last = logits[:, -1, :] / temperature
        probs = F.softmax(logits_last, dim=-1)

        # Sample from distribution
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    # Decode generated tokens
    output_text = tokenizer.decode(generated[0].tolist())
    return output_text


def generate_text_nnx(
    model,
    tokenizer,
    prompt: str,
    block_size: int,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> str:
    """Generate text from a trained NNX GPT model given a string prompt."""
    input_ids = jnp.array(tokenizer.encode(prompt).ids, dtype=jnp.int32)[None, :]
    key = jax.random.PRNGKey(42)

    for _ in range(max_new_tokens):
        x_cond = input_ids[:, -block_size:]
        logits = model(x_cond, training=False)
        logits_last = logits[:, -1, :] / temperature
        key, subkey = jax.random.split(key)
        next_token = jax.random.categorical(subkey, logits_last)
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)

    return tokenizer.decode(input_ids[0].tolist())


def generate_text_hf(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    device: str = "cpu",
    repetition_penalty: float = 1.3,
) -> str:
    """Generate text from a fine-tuned HuggingFace GPT-2 model."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def save_flax_model(
    params, mlflow_run_dir: str, filename: str = "model_params.msgpack"
):
    path = Path(mlflow_run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    mlflow.log_artifact(str(path))


def load_flax_model(model, artifact_path: str):
    with open(artifact_path, "rb") as f:
        param_bytes = f.read()
    return serialization.from_bytes(
        model.init(jax.random.PRNGKey(0), jnp.ones((1, 10))), param_bytes
    )
