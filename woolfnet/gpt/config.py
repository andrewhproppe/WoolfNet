"""
Small GPT-style Transformer for character/word-level language modeling.
Inspired by GPT-2 architecture.
"""

from dataclasses import dataclass
from pathlib import Path

from woolfnet.config import load_yaml


@dataclass
class GPTConfig:
    """
    Configuration for GPT-style model.
    """

    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "Embedding dim must be divisible by num_heads"

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        """
        Set the config attributes from a yaml file.
        """
        config = load_yaml(yaml_path)
        return GPTConfig(**config.model_params.to_dict())
