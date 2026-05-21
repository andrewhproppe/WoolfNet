"""Project high-dimensional embeddings to 2D via UMAP, with on-disk caching."""

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class UmapProjector:
    """UMAP projector with content-addressed disk caching."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        seed: int = 42,
        cache_dir: Path | None = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.seed = seed
        self.cache_dir = cache_dir

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Project ``matrix`` to ``n_components`` dims, using the disk cache when available."""
        cache_path = self._cache_path(matrix) if self.cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            logger.info(f"UMAP cache hit: {cache_path.name}")
            return np.load(cache_path)["embedding"]

        import umap

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.seed,
        )
        logger.info(
            f"Fitting UMAP on matrix of shape {matrix.shape} "
            f"(n_neighbors={self.n_neighbors}, min_dist={self.min_dist})"
        )
        result = reducer.fit_transform(matrix).astype(np.float32)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, embedding=result)
            logger.info(f"UMAP cached to {cache_path.name}")

        return result

    def _cache_path(self, matrix: np.ndarray) -> Path:
        h = hashlib.sha256()
        h.update(matrix.tobytes())
        h.update(
            f"{self.n_components}|{self.n_neighbors}|{self.min_dist}|"
            f"{self.metric}|{self.seed}".encode()
        )
        return self.cache_dir / f"umap_{h.hexdigest()[:16]}.npz"
