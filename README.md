# WoolfNet

A GPT-style decoder-only language model trained from scratch on the collected works of Virginia Woolf, sourced from [Project Gutenberg Australia](https://gutenberg.net.au/). Supports three model backends: PyTorch, JAX/Flax Linen (legacy), and JAX/Flax NNX (preferred).

## Architecture

The model is a standard GPT-2-style transformer:

- **Token embeddings** (no learned positional embeddings — uses RoPE)
- **N decoder blocks**: pre-norm → causal multi-head self-attention → SwiGLU feedforward (4× expansion)
- **Output**: LayerNorm → linear projection to vocab logits (weight-tied with token embedding)

Key hyperparameters (configurable via YAML):

| Parameter | Default |
|---|---|
| `vocab_size` | 5 000–16 000 |
| `block_size` | 128–256 |
| `n_layer` | 4–6 |
| `n_head` | 4–8 |
| `n_embd` | 256 |
| `dropout` | 0.05–0.30 |

Three parallel implementations live in `woolfnet/gpt/`:

| File | Framework | Status |
|---|---|---|
| `model.py` | PyTorch | Active |
| `model_jax.py` | JAX + Flax Linen | Legacy |
| `model_nnx.py` | JAX + Flax NNX | Preferred JAX backend |

## Setup

Requires Python 3.12+.

```bash
pip install -e .[test]
```

This installs the `woolf` CLI entry point.

## Full Pipeline

### 1. Download raw texts

```bash
woolf data download-raw
```

Downloads Woolf's works from Project Gutenberg Australia to `woolfnet/data/raw/`.

### 2. Clean raw texts

```bash
woolf data clean-raw-data
```

Outputs cleaned texts to `woolfnet/data/cleaned/`.

### 3. Build corpus

```bash
woolf data build-corpus --style both   # novel | essay | both
```

Outputs a single `.txt` corpus to `woolfnet/data/corpora/`.

### 4. Train BPE tokenizer

```bash
woolf tokenizer train-bpe \
  --corpus woolfnet/data/corpora/woolf_both_corpus.txt \
  --vocab-size 5000
```

Saves `vocab.json` + `merges.txt` to `woolfnet/data/tokenizers/woolf_both_corpus/`.

### 5. Prepare HDF5 dataset

```bash
woolf data prepare-dataset \
  --corpus woolfnet/data/corpora/woolf_both_corpus.txt \
  --tokenizer woolfnet/data/tokenizers/woolf_both_corpus \
  --block-size 128
```

Produces `woolfnet/data/datasets/woolf_both_corpus_dataset.hdf5` with `inputs` and `labels` tensors of shape `[num_blocks, block_size - 1]`.

### 6a. Train — PyTorch

```bash
woolf gpt train \
  --model-config woolfnet/gpt/configs/gpt_base.yml \
  --training-config woolfnet/gpt/configs/training_config.yml \
  --disable-logging
```

### 6b. Train — JAX NNX (preferred JAX backend)

```bash
woolf gpt train_jax_nnx \
  --config woolfnet/gpt/configs/training_config_jax.yml \
  --disable-logging
```

Omit `--disable-logging` to track metrics and artifacts with MLflow (stored locally at `woolfnet/data/mlruns`).

## Project Layout

```
woolfnet/
├── cli.py                    # Click CLI entry point
├── config.py                 # YAML config loader (attribute-accessible)
├── paths.py                  # ROOT_DIR, DATA_DIR, MLFLOW_LOCAL_URI
├── data/
│   ├── tasks.py              # download, clean, corpus build, dataset prep
│   ├── dataset.py            # CorpusDataset + DataLoader
│   └── utils.py
├── tokenization/
│   └── tasks.py              # BPE tokenizer training (HuggingFace tokenizers)
├── gpt/
│   ├── config.py             # GPTConfig dataclass
│   ├── model.py              # PyTorch GPT
│   ├── model_jax.py          # JAX/Flax Linen GPT (legacy)
│   ├── model_nnx.py          # JAX/Flax NNX GPT (preferred)
│   ├── training.py           # Training loop
│   ├── evaluation.py         # Perplexity + cross-entropy evaluation
│   ├── metrics.py            # Metric helpers
│   ├── tasks.py              # CLI task wrappers
│   ├── utils.py
│   ├── __init__.py           # LR_SCHEDULER_REGISTRY, ACTIVATION_REGISTRY, etc.
│   ├── configs/
│   │   ├── gpt_base.yml
│   │   ├── training_config.yml
│   │   └── training_config_jax.yml
│   └── artifacts/            # Saved model checkpoints
└── utils/
    ├── data.py
    ├── general.py
    └── mlflow.py
```

## Development

```bash
# Lint
ruff check .

# Format
ruff format .

# Tests
pytest
pytest woolfnet/gpt/tests/test_model.py   # single file
pytest -k test_gpt_model                  # single test
```

## Dependencies

Core: `torch`, `jax`, `flax`, `optax`, `tokenizers`, `h5py`, `mlflow`, `numpy`, `pandas`, `polars`.
