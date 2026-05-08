import logging
from pathlib import Path
from typing import Literal

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

from woolfnet.utils.data import numpy_collate

logger = logging.getLogger(__name__)


class CorpusDataset:
    """
    Load the dataset file from h5py and create torch DataLoader objects with optional numpy
    collation function for JAX. Supports an optional held-out validation split (last val_split
    fraction of the corpus, taken contiguously to respect temporal ordering of the text blocks).
    """

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int,
        collate_for: Literal["torch", "jax"] = "jax",
        val_split: float = 0.0,
    ):
        self.batch_size = batch_size
        self.collate_for = collate_for

        with h5py.File(dataset_path, mode="r") as h5file:
            inputs = torch.tensor(h5file["inputs"][:], dtype=torch.long)
            labels = torch.tensor(h5file["labels"][:], dtype=torch.long)

        n = len(inputs)
        n_val = int(n * val_split)
        n_train = n - n_val

        self._train = TensorDataset(inputs[:n_train], labels[:n_train])
        self._val = TensorDataset(inputs[n_train:], labels[n_train:]) if n_val > 0 else None

    def _make_loader(self, dataset: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=numpy_collate if self.collate_for == "jax" else None,
        )

    def create_dataloader(self) -> DataLoader:
        return self._make_loader(self._train, shuffle=True)

    def create_val_dataloader(self) -> DataLoader:
        if self._val is None:
            raise ValueError("No validation split configured (val_split=0.0).")
        return self._make_loader(self._val, shuffle=False)


class GPT2CorpusDataset:
    """
    Tokenizes a raw text corpus with a HuggingFace tokenizer and creates non-overlapping
    blocks of a fixed size, suitable for GPT-2 fine-tuning. Supports an optional held-out
    validation split (last val_split fraction, contiguous).
    """

    def __init__(
        self, corpus_path: Path, tokenizer, block_size: int, batch_size: int, val_split: float = 0.0
    ):
        self.batch_size = batch_size
        text = corpus_path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text)
        blocks = [
            torch.tensor(token_ids[i : i + block_size], dtype=torch.long)
            for i in range(0, len(token_ids) - block_size, block_size)
        ]
        n = len(blocks)
        n_val = int(n * val_split)
        n_train = n - n_val
        self._train = blocks[:n_train]
        self._val = blocks[n_train:] if n_val > 0 else None
        logger.info(f"GPT2CorpusDataset: {n_train} train / {n_val} val blocks (block={block_size})")

    def create_dataloader(self) -> DataLoader:
        """Return a shuffled DataLoader over the training split."""
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True)

    def create_val_dataloader(self) -> DataLoader:
        """Return an unshuffled DataLoader over the validation split."""
        if self._val is None:
            raise ValueError("No validation split configured (val_split=0.0).")
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False)
