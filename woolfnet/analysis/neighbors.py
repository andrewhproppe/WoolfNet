"""Cosine-similarity nearest-neighbor search over a token-embedding set."""

import logging

import numpy as np

from woolfnet.analysis.embeddings import EmbeddingSet

logger = logging.getLogger(__name__)


class NeighborSearch:
    """Cosine top-k lookup over the embeddings in an ``EmbeddingSet``.

    Pre-normalises the matrix on init so each query is a single matrix-vector product.
    """

    def __init__(self, embedding_set: EmbeddingSet):
        self.tokens = embedding_set.tokens
        self.token_to_idx: dict[str, int] = {t: i for i, t in enumerate(embedding_set.tokens)}
        norms = np.linalg.norm(embedding_set.matrix, axis=1, keepdims=True)
        self.unit_matrix: np.ndarray = embedding_set.matrix / np.clip(norms, 1e-12, None)

    def top_k_for_token(self, token: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k nearest tokens to ``token`` by cosine similarity, excluding ``token`` itself."""
        idx = self.token_to_idx.get(token)
        if idx is None:
            return []
        sims = self.unit_matrix @ self.unit_matrix[idx]
        sims[idx] = -np.inf
        return self._take_top_k(sims, k)

    def top_k_for_vector(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Top-k nearest tokens to an arbitrary query vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        sims = self.unit_matrix @ (vector / norm)
        return self._take_top_k(sims, k)

    def _take_top_k(self, sims: np.ndarray, k: int) -> list[tuple[str, float]]:
        top_indices = np.argpartition(-sims, k)[:k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        return [(self.tokens[i], float(sims[i])) for i in top_indices]
