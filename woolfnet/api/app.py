"""FastAPI server exposing the configured WoolfNet inference models."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException

from woolfnet.analysis import (
    NeighborSearch,
    UmapProjector,
    extract_token_embeddings,
    fit_clusterer,
)
from woolfnet.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    ModelsResponse,
    NeighborInfo,
    NeighborsRequest,
    NeighborsResponse,
    ProjectionResponse,
)
from woolfnet.inference import WoolfModel
from woolfnet.paths import DATA_DIR, ROOT_DIR

logger = logging.getLogger(__name__)

CORPUS_PATH = ROOT_DIR / "data" / "corpora" / "woolf_both_corpus.txt"
EMBEDDING_CACHE_DIR = DATA_DIR / "embeddings"
TOP_N_TOKENS = 5000

_LOADED: dict[str, WoolfModel] = {}


@dataclass
class _EmbeddingState:
    tokens: list[str]
    matrix: np.ndarray
    projections: dict[int, np.ndarray]
    cluster_layers: list[np.ndarray]
    membership_layers: list[np.ndarray]
    neighbors: NeighborSearch
    model: WoolfModel


_EMBEDDING_STATE: dict[str, _EmbeddingState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log available models on startup; drop cached models on shutdown."""
    logger.info(f"Starting WoolfNet API. Available models: {WoolfModel.available()}")
    yield
    _LOADED.clear()
    _EMBEDDING_STATE.clear()


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


def _get_embedding_state(name: str) -> _EmbeddingState:
    if name not in _EMBEDDING_STATE:
        model = _get_model(name)
        es = extract_token_embeddings(model, top_n=TOP_N_TOKENS, corpus_path=CORPUS_PATH)
        clusterer = fit_clusterer(es.matrix)
        cluster_layers = [np.asarray(layer) for layer in clusterer.cluster_layers_]
        membership_layers = [np.asarray(m) for m in clusterer.membership_strength_layers_]
        _EMBEDDING_STATE[name] = _EmbeddingState(
            tokens=es.tokens,
            matrix=es.matrix,
            projections={},
            cluster_layers=cluster_layers,
            membership_layers=membership_layers,
            neighbors=NeighborSearch(es),
            model=model,
        )
        logger.info(
            f"Built embedding state for {name}: {es.matrix.shape}, "
            f"{len(cluster_layers)} cluster layer(s) "
            f"(sizes: {[len(set(layer.tolist()) - {-1}) for layer in cluster_layers]})"
        )
    return _EMBEDDING_STATE[name]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    """List configured models and whether they're currently loaded."""
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


@app.get("/embeddings/projection", response_model=ProjectionResponse)
def embedding_projection(model: str, dim: int = 2) -> ProjectionResponse:
    """Return tokens, UMAP projection, and KMeans cluster ids for one model."""
    if dim not in (2, 3):
        raise HTTPException(status_code=400, detail=f"dim must be 2 or 3, got {dim}")
    state = _get_embedding_state(model)
    if dim not in state.projections:
        state.projections[dim] = UmapProjector(
            n_components=dim, cache_dir=EMBEDDING_CACHE_DIR
        ).fit_transform(state.matrix)
    return ProjectionResponse(
        tokens=state.tokens,
        projection=state.projections[dim].tolist(),
        cluster_layers=[layer.tolist() for layer in state.cluster_layers],
        membership_layers=[m.tolist() for m in state.membership_layers],
    )


@app.post("/embeddings/neighbors", response_model=NeighborsResponse)
def embedding_neighbors(req: NeighborsRequest) -> NeighborsResponse:
    """Return the nearest tokens to the user's query, per BPE piece."""
    state = _get_embedding_state(req.model)
    text = req.query if req.query.startswith(" ") else " " + req.query
    if state.model.source == "torch":
        ids = state.model.tokenizer.encode(text).ids
        pieces = [state.model.tokenizer.id_to_token(i) for i in ids]
    else:
        ids = state.model.tokenizer.encode(text)
        pieces = [state.model.tokenizer.convert_ids_to_tokens(i) for i in ids]
    neighbors_per_piece = {
        piece: [
            NeighborInfo(token=t, similarity=s)
            for t, s in state.neighbors.top_k_for_token(piece, k=req.k)
        ]
        for piece in pieces
    }
    return NeighborsResponse(
        tokenized_pieces=pieces,
        neighbors_per_piece=neighbors_per_piece,
    )
