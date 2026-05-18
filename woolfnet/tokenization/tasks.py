import logging
from datetime import datetime
from pathlib import Path

import click
from tokenizers import ByteLevelBPETokenizer

from woolfnet.paths import DATA_DIR

logger = logging.getLogger(__name__)


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
    help="Vocabulary size.",
)
@click.option(
    "--name",
    type=str,
    help="Optional additional naming string for the tokenizer.",
)
def train_bpe(corpus: Path, vocab_size: int, name: str):
    name_str = name if name is not None else ""
    tokenizer_dir = Path(
        DATA_DIR / "tokenizers" / (corpus.parts[-1].split(".txt")[0] + name_str)
    )
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train the tokenizer
    tokenizer.train(
        files=[str(corpus)],
        vocab_size=vocab_size,  # small corpus, small vocab is enough
        min_frequency=2,  # Ignore rare tokens
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    # Save tokenizer files
    tokenizer.save_model(str(tokenizer_dir))

    metadata = {"corpus": corpus, "vocab_size": vocab_size, "dt": datetime.now()}
    print(f"Tokenizer trained and saved to {tokenizer_dir}")


COMMANDS = {
    "train-bpe": train_bpe,
}
