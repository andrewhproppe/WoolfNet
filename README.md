# WoolfNet

A small end-to-end project for training, fine-tuning, and serving GPT-style language models on the collected works of Virginia Woolf. Texts are sourced from [Project Gutenberg Australia](https://gutenberg.net.au/).

WoolfNet was originally a personal NLP/JAX learning project. It now wraps three model backends behind a FastAPI service and a Gradio UI, and ships with a Docker setup for reproducible deployment.

## What's in the box

Three model backends are exposed via the same API:

| Name           | Backend                              | Source                                                     |
| -------------- | ------------------------------------ | ---------------------------------------------------------- |
| `woolf-scratch`| PyTorch GPT, decoder-only, ~10M params | Trained from scratch on the Woolf corpus (`model.pt`)    |
| `gpt2`         | HuggingFace GPT-2 small (~124M)      | Pretrained, no fine-tuning                                 |
| `gpt2-woolf`   | HuggingFace GPT-2 small fine-tuned   | Fine-tuned on the Woolf corpus (`gpt2_finetuned/`)         |

`woolf-scratch` is the small from-scratch model — interesting to read, weak at coherence. `gpt2` is a fluent baseline. `gpt2-woolf` is the most fun: GPT-2's prior with Woolf's vocabulary and cadence on top.

The codebase also keeps two JAX implementations of the same scratch architecture (Flax Linen, legacy; Flax NNX, preferred). The HTTP API exposes only the PyTorch path; JAX training is supported via the CLI.

## Architecture

```
woolfnet/
├── cli.py                 # Click CLI: data, tokenizer, gpt, serve, ui
├── config.py              # YAML loader (DotDict, dot-access)
├── paths.py
├── api/                   # FastAPI inference server
│   ├── app.py             # /health, /models, /generate
│   └── schemas.py
├── app/                   # Gradio frontend (calls the API over HTTP)
│   ├── ui.py
│   └── tasks.py           # `woolf serve`, `woolf ui` CLI entrypoints
├── inference/
│   └── loader.py          # `load_backend(name) -> Backend`
├── configs/inference.yml  # API/UI host + port defaults
├── data/                  # download, clean, corpus, dataset prep
├── tokenization/          # BPE tokenizer training
└── gpt/
    ├── model.py           # PyTorch GPT (used by API)
    ├── model_jax.py       # Flax Linen GPT (legacy)
    ├── model_nnx.py       # Flax NNX GPT (preferred JAX backend)
    ├── training.py        # training + fine-tuning loops
    ├── evaluation.py      # perplexity / BPC / Distinct-N comparison
    └── artifacts/         # checkpoints (model.pt, gpt2_finetuned/, ...)
```

Generation logic lives in `gpt/utils.py` and is shared by training (sample previews) and inference (the API).

## Setup

Python 3.12+.

```bash
pip install -e .[test]
```

This installs the `woolf` CLI.

## Training pipeline

```bash
woolf data download-raw                                  # Project Gutenberg AU → data/raw/
woolf data clean-raw-data                                # → data/cleaned/
woolf data build-corpus --style both                     # → data/corpora/woolf_both_corpus.txt
woolf tokenizer train-bpe \
  --corpus data/corpora/woolf_both_corpus.txt \
  --vocab-size 16000                                     # → data/tokenizers/woolf_both_corpus/
woolf data prepare-dataset \
  --corpus data/corpora/woolf_both_corpus.txt \
  --tokenizer data/tokenizers/woolf_both_corpus \
  --block-size 256                                       # → data/datasets/*.hdf5
```

CLI paths are relative to the working directory; run from `woolfnet/` (per the existing convention) or pass absolute paths.

### Train scratch model (PyTorch)

```bash
woolf gpt train \
  --model-config gpt/configs/gpt_base.yml \
  --training-config gpt/configs/training_config.yml
```

### Fine-tune GPT-2 on Woolf

```bash
woolf gpt finetune --config gpt/configs/training_config_finetune.yml
```

Drop `--disable-logging` to track runs in local MLflow (`data/mlruns/`). With logging on, training artifacts (model weights + tokenizer + GPT config for scratch runs; the full HuggingFace `save_pretrained` directory for fine-tunes) are uploaded as MLflow artifacts. The inference layer can then load them by run ID or versioned name — see [`woolfnet/configs/inference.yml`](woolfnet/configs/inference.yml).

### Compare models

```bash
woolf gpt evaluate \
  --our-model-weights gpt/artifacts/model_weights/model.pt \
  --finetuned-model gpt/artifacts/gpt2_finetuned/<run>/
```

Reports perplexity, bits-per-character, and Distinct-1/2 lexical diversity.

## Serving — FastAPI

```bash
woolf serve            # listens on :8000
```

Endpoints:
- `GET /health` — liveness.
- `GET /models` — list backends and load state.
- `POST /generate` — generate text.

```bash
curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Mrs Dalloway said", "model": "gpt2-woolf", "max_new_tokens": 60, "temperature": 0.8}'
```

```json
{
  "model": "gpt2-woolf",
  "prompt": "Mrs Dalloway said",
  "generated_text": "Mrs Dalloway said she would buy the flowers herself. She stopped in front of her little bedroom at dusk..."
}
```

Models are loaded lazily on first request and cached for the lifetime of the process. `temperature` is clamped to `(0, 2]` and `max_new_tokens` to `[1, 512]`.

## UI — Gradio

```bash
woolf ui               # listens on :7860, calls WOOLFNET_API_URL (default http://localhost:8000)
```

Prompt box, model dropdown, temperature + max-tokens sliders. Talks to the FastAPI backend over HTTP so the two are independently scalable / restartable.

## Docker

```bash
docker compose up --build
```

Two services come up:

- `api` on `http://localhost:8000`
- `ui`  on `http://localhost:7860`

Volumes mount `gpt/artifacts` and `data/tokenizers` from the host so checkpoints persist across rebuilds. The HuggingFace cache is on a named volume to avoid re-downloading `gpt2` on every restart.

CPU-only torch is installed inside the image to keep size reasonable (~1.5 GB instead of ~3 GB). For GPU use, swap the torch wheel in the Dockerfile.

## Example outputs

Prompt: *"Mrs Dalloway said she would buy the flowers herself."*

- **gpt2-woolf**: *"She stopped in front of her little bedroom at dusk and turned to Mrs Riddell, who had already opened a door on top of it without him noticing; then suddenly there was nothing else but white..."*
- **woolf-scratch**: *"Mrs. Dalloway was at any rate the same. She would have been doubtful whether, from all the evening it was, how she would be killed in them..."*

Both produced at `temperature=0.8`. The scratch model is recognisably Woolf-tinted but locally incoherent; the fine-tuned GPT-2 carries the prior fluency forward.

## Development

```bash
ruff check .
ruff format .
pytest
```

## Future work

- Beam search / top-k / top-p in the API (currently temperature-only sampling).
- Streaming generation (`text/event-stream`).
- A second fine-tune at higher epoch count for a stronger `gpt2-woolf`.
- Replace the legacy Flax Linen path with a single NNX/PyTorch story.
