"""
Utility functions for gathering, cleaning, and assembling data.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from woolfnet.paths import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class Book:
    """
    Data class for handling the retrieval and cleaning of book texts.
    """

    # Book title
    title: str

    # Download link
    url: str

    # Starting (ending) lines used to index and slice the text to trim off any
    # text that is not part of the book
    start_line: str
    end_line: str

    # 'novel' or 'essay' - for building corpora that contain either or both
    style: str


WOOLF_BOOKS = [
    Book(
        title="Monday or Tuesday",
        url="https://www.gutenberg.net.au/ebooks02/0200211.txt",
        start_line="Whatever hour you woke there was a door shunting",
        end_line="Ah, the mark on the wall! It was a snail.",
        style="novel",
    ),
    Book(
        title="Mrs Dalloway",
        url="https://www.gutenberg.net.au/ebooks02/0200991.txt",
        start_line="Mrs. Dalloway said she would buy the flowers herself.",
        end_line="For there she was.",
        style="novel",
    ),
    Book(
        title="To the Lighthouse",
        url="https://www.gutenberg.net.au/ebooks01/0100101.txt",
        start_line="Yes, of course, if it's fine tomorrow,",
        end_line="I have had my vision.",
        style="novel",
    ),
    Book(
        title="Orlando",
        url="https://www.gutenberg.net.au/ebooks02/0200331.txt",
        start_line="Many friends have helped me in writing this book",
        end_line="Thursday, the eleventh of October, Nineteen hundred and Twenty\nEight.",
        style="novel",
    ),
    Book(
        title="A Room of One's Own",
        url="https://www.gutenberg.net.au/ebooks02/0200791.txt",
        start_line="But, you may say, we asked you to speak about women",
        end_line="and that so to work, even in poverty\nand obscurity, is worth while.",
        style="essay",
    ),
    Book(
        title="The Waves",
        url="https://www.gutenberg.net.au/ebooks02/0201091.txt",
        start_line="The sun had not yet risen.",
        end_line="The waves broke on the shore.",
        style="novel",
    ),
    Book(
        title="Three Guineas",
        url="https://www.gutenberg.net.au/ebooks02/0200931.txt",
        start_line="Three years is a long time to leave a letter unanswered",
        end_line="Histoire de ma Vie, by George Sand",
        style="essay",
    ),
    Book(
        title="Between the Acts",
        url="https://www.gutenberg.net.au/ebooks03/0301171.txt",
        start_line="It was a summer's night and they were talking",
        end_line="Then the curtain rose.  They spoke.",
        style="novel",
    ),
    Book(
        title="The Years",
        url="https://www.gutenberg.net.au/ebooks03/0301221.txt",
        start_line="It was an uncertain spring.",
        end_line="The sun had risen, and the sky above the houses wore an air of\nextraordinary beauty, simplicity and peace.",
        style="novel",
    ),
    Book(
        title="Collected Essays",
        url="https://www.gutenberg.net.au/ebooks02/0200771.txt",
        start_line="There is a sentence in Dr. Johnson's Gray which might well be",
        end_line=" And now, in the shadowed half of the world, to sleep.",
        style="essay",
    ),
    Book(
        title="Collected Short Stories",
        url="https://www.gutenberg.net.au/ebooks02/0200781.txt",
        start_line="Perhaps it was the middle of January in the present that",
        end_line="startled up into the air by a stone thrown at it.",
        style="novel",
    ),
    Book(
        title="The Voyage Out",
        url="https://www.gutenberg.net.au/ebooks/m00020.txt",
        start_line="As the streets that lead from the Strand to the Embankment",
        end_line="and passing him one after another on their way\nto bed",
        style="novel",
    ),
    Book(
        title="Night and Day",
        url="https://www.gutenberg.net.au/ebooks/m00019.txt",
        start_line="It was a Sunday evening in October",
        end_line="she murmured back to him.",
        style="novel",
    ),
    Book(
        title="Jacob's Room",
        url="https://www.gutenberg.net.au/ebooks/m00018.txt",
        start_line="wrote Betty Flanders, pressing her heels rather deeper",
        end_line="She held out a pair of Jacob's old shoes.",
        style="novel",
    ),
]


def download_raw_text() -> None:
    """
    Download the raw texts from Project Gutenberg Australia. Compared to the US and
    Canadian versions, Project Gutenberg Australia has the most comprehensive collec-
    tion of Woolf's works.
    """
    data_dest = DATA_DIR / "raw"
    data_dest.mkdir(parents=True, exist_ok=True)
    for book in tqdm(WOOLF_BOOKS, desc="Downloading from Project Gutenberg Australia"):
        title = book.title
        url = book.url
        try:
            text = requests.get(url).text
            fname = data_dest / f"{title}.txt"
            if not Path(fname).exists():
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(text)
        except Exception as e:
            logger.info(f"Error when trying to donwload title {title}")
            logger.debug(e)


def clean_raw_text() -> None:
    """
    Remove preamble and ending sections from the raw text files from Project Gutenberg.
    The contents  of these sections are specific to Project Gutenberg Australia, but
    are not consistent - so we use the manually defined start and end lines of each
    book to index and trim the text.
    """
    data_src = DATA_DIR / "raw"
    output_dir = DATA_DIR / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    assert data_src.exists(), f"No raw data directory found at {data_src}"

    for file in data_src.glob("*.txt"):
        current_book = None
        # Find the matching Book object
        title = file.parts[-1].split(".txt")[0]
        for book in WOOLF_BOOKS:
            if book.title == title:
                current_book = book
                break
        assert current_book, f"No matching Book object found for title {title}"

        text = file.read_text(encoding="utf-8", errors="ignore")

        assert (
            text.count(current_book.start_line) != 0
        ), f"Start line '{current_book.start_line}' not found for {title}"

        assert (
            text.count(current_book.end_line) != 0
        ), f"Ending line '{current_book.end_line}' not found for {title}"

        # Then ensure we have only one instance of the ending line
        assert (
            text.count(current_book.end_line) == 1
        ), f"Found multiple instances of end line {current_book.end_line} in the text. A more specific ending line needs to be provided."

        start_idx = text.find(current_book.start_line)

        # Include the length of the ending line in the indexing
        end_idx = text.find(current_book.end_line) + len(book.end_line)

        text = text[start_idx:end_idx]

        # Basic cleanup
        text = re.sub(r"\s+", " ", text)  # collapse whitespace
        text = text.strip()

        outpath = output_dir / file.name
        outpath.write_text(text, encoding="utf-8")
        logger.info(f"Cleaned {file.name} → {len(text) // 1000} KB")
