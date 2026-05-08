"""NLP evaluation metrics for comparing our from-scratch GPT vs fine-tuned GPT-2."""

import logging
import math

import torch
from tokenizers import ByteLevelBPETokenizer
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

EVAL_PROMPTS = [
    "Mrs. Dalloway said she would buy the flowers herself.",
    "She thought of the room, the flowers,",
    "Nothing was simply one thing.",
    "Time passes. The nights are dark,",
    "I thought how unpleasant it is to be locked out;",
    "The waves broke on the shore.",
]


def _avg_ce_torch(model, tokens: list[int], block_size: int, device: str) -> float:
    """Average CE loss (nats/token) for our PyTorch GPT model."""
    model.eval()
    total_loss, n_blocks = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(tokens) - block_size, block_size):
            x = torch.tensor(tokens[i : i + block_size], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(tokens[i + 1 : i + block_size + 1], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            total_loss += CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            n_blocks += 1
    return total_loss / n_blocks if n_blocks > 0 else float("inf")


def _avg_ce_hf(model, tokens: list[int], block_size: int, device: str) -> float:
    """Average CE loss (nats/token) for a HuggingFace causal LM."""
    model.eval()
    total_loss, n_blocks = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(tokens) - block_size, block_size):
            block = torch.tensor(tokens[i : i + block_size], dtype=torch.long, device=device).unsqueeze(0)
            total_loss += model(input_ids=block, labels=block).loss.item()
            n_blocks += 1
    return total_loss / n_blocks if n_blocks > 0 else float("inf")


def compute_bpc(avg_ce_nats: float, text: str, n_tokens: int) -> float:
    """Convert avg CE loss (nats/token) to bits per character — comparable across tokenizers."""
    chars_per_token = len(text) / n_tokens
    return avg_ce_nats / (chars_per_token * math.log(2))


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-N: fraction of unique word n-grams across all texts. Higher = more diverse."""
    all_ngrams, unique_ngrams = [], set()
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / len(all_ngrams) if all_ngrams else 0.0


def run_comparison(
    our_model,
    our_tokenizer: ByteLevelBPETokenizer,
    our_block_size: int,
    hf_model,
    hf_tokenizer,
    val_text: str,
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> dict:
    """
    Evaluate both models on BPC, perplexity, and Distinct-N. Returns a results dict
    with per-metric scores and generated samples for each prompt in EVAL_PROMPTS.
    """
    from woolfnet.gpt.utils import generate_text, generate_text_hf

    our_tokens = our_tokenizer.encode(val_text).ids
    hf_tokens = hf_tokenizer.encode(val_text)

    logger.info("Computing CE loss for our model...")
    our_ce = _avg_ce_torch(our_model, our_tokens, our_block_size, device)
    logger.info("Computing CE loss for fine-tuned GPT-2...")
    hf_ce = _avg_ce_hf(hf_model, hf_tokens, 512, device)

    our_bpc = compute_bpc(our_ce, val_text, len(our_tokens))
    hf_bpc = compute_bpc(hf_ce, val_text, len(hf_tokens))

    logger.info("Generating samples from both models...")
    our_samples = [
        generate_text(our_model, our_tokenizer, p, max_new_tokens, temperature, device)
        for p in EVAL_PROMPTS
    ]
    hf_samples = [
        generate_text_hf(hf_model, hf_tokenizer, p, max_new_tokens, temperature, device)
        for p in EVAL_PROMPTS
    ]

    return {
        "perplexity": {"our": math.exp(our_ce), "gpt2": math.exp(hf_ce)},
        "bpc": {"our": our_bpc, "gpt2": hf_bpc},
        "distinct_1": {"our": distinct_n(our_samples, 1), "gpt2": distinct_n(hf_samples, 1)},
        "distinct_2": {"our": distinct_n(our_samples, 2), "gpt2": distinct_n(hf_samples, 2)},
        "samples": {"prompts": EVAL_PROMPTS, "our": our_samples, "gpt2": hf_samples},
    }
