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

# CPU-only torch
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY pyproject.toml ./
COPY woolfnet ./woolfnet

RUN pip install -e .

EXPOSE 8000 7860

CMD ["woolf", "serve", "--host", "0.0.0.0", "--port", "8000"]
