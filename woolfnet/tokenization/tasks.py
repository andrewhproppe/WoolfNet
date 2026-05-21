"""CLI task for training a byte-level BPE tokenizer on a corpus."""

import json
import logging
from datetime import datetime
from pathlib import Path

import click
from tokenizers import ByteLevelBPETokenizer

from woolfnet.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Defaults for tokenization
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


@click.command()
@click.option(
    "--corpus",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the corpus to be used for training the BPE",
)
@click.option(
    "--vocab-size",
    type=int,
    default=16000,
    show_default=True,
    help="Vocabulary size.",
)
@click.option(
    "--name",
    type=str,
    help="Optional additional naming string for the tokenizer.",
)
def train_bpe(corpus: Path, vocab_size: int, name: str):
    """Train a byte-level BPE tokenizer on ``corpus`` and save it under data/tokenizers/."""
    name_str = name if name is not None else ""
    tokenizer_dir = Path(DATA_DIR / "tokenizers" / (corpus.parts[-1].split(".txt")[0] + name_str))
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus)],
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(tokenizer_dir))

    metadata = {
        "corpus": str(corpus),
        "vocab_size": vocab_size,
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    (tokenizer_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"Tokenizer trained and saved to {tokenizer_dir}")


COMMANDS = {
    "train-bpe": train_bpe,
}
