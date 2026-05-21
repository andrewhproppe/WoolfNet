"""Extract token embeddings from a loaded ``WoolfModel``."""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from woolfnet.inference import WoolfModel

logger = logging.getLogger(__name__)


def fit_clusterer(matrix: np.ndarray, seed: int = 42):
    """Fit an EVoC clusterer on ``matrix`` and return it.

    The returned object exposes ``cluster_layers_`` (multi-granularity labels;
    noise = -1), ``membership_strength_layers_`` (per-token confidence, 0-1, per
    layer), ``cluster_tree_`` (hierarchy), and ``duplicates_`` (near-duplicate sets).
    """
    import evoc

    return evoc.EVoC(random_state=seed).fit(matrix)


def display_token(token: str, leading_space: str = "") -> str:
    """BPE token rendered for display: ``Ġ`` becomes ``leading_space`` (default empty)."""
    if token.startswith("Ġ"):
        return leading_space + token[1:]
    return token


@dataclass
class EmbeddingSet:
    """A model's token embeddings paired with their token strings."""

    model_name: str
    tokens: list[str]
    matrix: np.ndarray


def extract_token_embeddings(
    model: WoolfModel,
    top_n: int | None = None,
    corpus_path: Path | None = None,
) -> EmbeddingSet:
    """Token embedding matrix for ``model``, optionally restricted to the ``top_n``
    most frequent tokens in ``corpus_path``."""
    import torch

    if model.source == "torch":
        weights = model.model.token_embedding.weight
        vocab = model.tokenizer.get_vocab()
        tokens: list[str] = [""] * len(vocab)
        for tok, idx in vocab.items():
            tokens[idx] = tok
    elif model.source == "huggingface":
        weights = model.model.transformer.wte.weight
        tokens = [model.tokenizer.convert_ids_to_tokens(i) for i in range(len(model.tokenizer))]
    else:
        raise ValueError(f"Unsupported model source for embedding extraction: {model.source}")
    matrix = weights.detach().to(torch.float32).cpu().numpy()

    if top_n is None or top_n >= len(tokens):
        return EmbeddingSet(model_name=model.name, tokens=tokens, matrix=matrix)

    if corpus_path is None:
        raise ValueError("corpus_path is required when top_n is set.")

    text = corpus_path.read_text(encoding="utf-8")
    if model.source == "torch":
        ids = model.tokenizer.encode(text).ids
    else:
        ids = model.tokenizer.encode(text)
    counts = Counter(ids)
    logger.info(
        f"{model.name}: {len(counts)} unique tokens in corpus "
        f"(of {len(tokens)} in vocab), keeping top {top_n}"
    )
    keep_indices = [tok_id for tok_id, _ in counts.most_common(top_n)]
    return EmbeddingSet(
        model_name=model.name,
        tokens=[tokens[i] for i in keep_indices],
        matrix=matrix[keep_indices],
    )
