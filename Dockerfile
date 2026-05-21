FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY woolfnet ./woolfnet

# Single resolve pass with the CPU torch index preferred. Pip never reaches for
# the PyPI CUDA build of torch, so the +cpu wheel is installed exactly once and
# no nvidia-* packages ride along.
RUN pip install -e . \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

EXPOSE 8000 7860

CMD ["woolf", "serve", "--host", "0.0.0.0", "--port", "8000"]
