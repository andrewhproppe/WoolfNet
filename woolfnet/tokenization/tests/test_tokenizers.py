"""
Unit tests for tokenizers and tokenization functions.
"""

from tokenizers import ByteLevelBPETokenizer

from woolfnet.paths import DATA_DIR


def test_bpe_tokenizer():
    """
    Test the encoding and decoding of the BPE tokenizer.
    """
    tokenizer_path = DATA_DIR / "tokenizers" / "bpe_test_tokenizer"
    tokenizer = ByteLevelBPETokenizer(
        str(tokenizer_path / "vocab.json"), str(tokenizer_path / "merges.txt")
    )
    sample_text = "Mrs. Dalloway said she would buy the flowers herself."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded.ids)
    assert sample_text == decoded, "Mismatch between sample text and decoded encoding."
