"""FastAPI server exposing the configured WoolfNet inference models."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from woolfnet.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    ModelsResponse,
)
from woolfnet.inference import WoolfModel

logger = logging.getLogger(__name__)

# Loaded models are cached for the lifetime of the app — each is several hundred MB.
_LOADED: dict[str, WoolfModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log available models on startup; drop cached models on shutdown."""
    logger.info(f"Starting WoolfNet API. Available models: {WoolfModel.available()}")
    yield
    _LOADED.clear()


app = FastAPI(
    title="WoolfNet Inference API",
    description="Generate text in the style of Virginia Woolf from the configured models.",
    version="0.1",
    lifespan=lifespan,
)


def _get_model(name: str) -> WoolfModel:
    if name not in _LOADED:
        try:
            _LOADED[name] = WoolfModel(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _LOADED[name]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    """List available models and whether they're currently loaded in memory."""
    descriptions = WoolfModel.descriptions()
    infos = [
        ModelInfo(name=name, description=descriptions[name], loaded=name in _LOADED)
        for name in WoolfModel.available()
    ]
    return ModelsResponse(models=infos)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text from the requested model."""
    model = _get_model(req.model)
    try:
        text = model.generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    return GenerateResponse(model=req.model, prompt=req.prompt, generated_text=text)
