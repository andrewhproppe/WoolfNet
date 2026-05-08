import logging
from pathlib import Path
from typing import List

import click
import h5py
import torch
from tokenizers import ByteLevelBPETokenizer

from woolfnet.data.utils import WOOLF_BOOKS, clean_raw_text, download_raw_text
from woolfnet.paths import DATA_DIR
from woolfnet.utils.general import debug_option

logger = logging.getLogger(__name__)


@click.command()
@debug_option
def download_raw() -> None:
    """
    Download the raw texts from Project Gutenberg Australia. Compared to the US and
    Canadian versions, Project Gutenberg Australia has the most comprehensive collec-
    tion of Woolf's works.
    """
    download_raw_text()


@click.command()
@debug_option
def clean_raw_data() -> None:
    clean_raw_text()


@click.command()
@debug_option
@click.option(
    "--style",
    type=click.Choice(choices=["novel", "essay", "both"]),
    default="both",
    help="Styles of books to include in the final corpus file.",
)
def build_corpus(style: List[str]) -> None:
    """
    Concatenate together texts from different books to form a corpus.
    """
    output_dir = DATA_DIR / "corpora"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fname = output_dir / f"woolf_{style}_corpus.txt"

    with open(output_fname, "w", encoding="utf-8") as f_out:
        for book in WOOLF_BOOKS:
            # Select either novels or essays (or both)
            if book.style == style or style == "both":
                file = DATA_DIR / "cleaned" / f"{book.title}.txt"
                try:
                    text = file.read_text(encoding="utf-8", errors="ignore")
                    f_out.write(text)
                    f_out.write("\n\n")
                except Exception as e:
                    logger.info(f"Error trying to add title '{book.title}' to the corpus")
                    logger.debug(e)

    logger.info(f"Corpus with style(s) '{style}' written to {output_fname}.")


@click.command()
@debug_option
@click.option(
    "--corpus",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the corpus to be converted into tokenized dataset",
)
@click.option(
    "--tokenizer",
    type=click.Path(path_type=Path),
    help="Tokenizer to use for the dataset preparation.",
)
@click.option(
    "--block-size",
    type=int,
    default=128,
    help="Token block size for model training",
)
@click.option(
    "--stride",
    type=int,
    default=None,
    help="Stride for overlapping windows. Defaults to block-size (no overlap).",
)
def prepare_dataset(corpus: Path, tokenizer: Path, block_size: int, stride: int) -> None:
    """
    Encode a corpus into tokens and save the dataset as torch tensors.
    """
    stride = stride if stride is not None else block_size

    # Load the tokenizer
    tokenizer = ByteLevelBPETokenizer(str(tokenizer / "vocab.json"), str(tokenizer / "merges.txt"))

    # Load in the corpus text
    text = corpus.read_text(encoding="utf-8")

    # Convert text into tokens
    logger.info("Encoding corpus into tokens")
    tokens = tokenizer.encode(text).ids

    tokens_tensor = torch.tensor(tokens, dtype=torch.long)

    # Create dataset: overlapping windows of block_size with given stride
    blocks = [
        tokens_tensor[i : i + block_size] for i in range(0, len(tokens_tensor) - block_size, stride)
    ]
    num_blocks = len(blocks)
    dataset = torch.stack(blocks)
    logger.info(f"Created {num_blocks} blocks (block_size={block_size}, stride={stride})")

    # Create inputs and labels for autoregressive model training
    inputs = dataset[:, :-1]
    labels = dataset[:, 1:]

    output_path = DATA_DIR / "datasets" / f"{corpus.parts[-1].split('.txt')[0]}_dataset.hdf5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: Add metadata tracking that includes which tokenizer was used

    # Save to .h5
    with h5py.File(output_path, "w") as h5file:
        h5file["inputs"] = inputs
        h5file["labels"] = labels

    logger.info(f"Saved dataset to {str(output_path)}")


COMMANDS = {
    "download-raw": download_raw,
    "clean-raw-data": clean_raw_data,
    "build-corpus": build_corpus,
    "prepare-dataset": prepare_dataset,
}
