"""Pydantic schemas for the inference API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    model: str = Field("gpt2-woolf", description="Backend name from /models.")
    max_new_tokens: int = Field(60, ge=1, le=512)
    temperature: float = Field(0.8, gt=0.0, le=2.0)


class GenerateResponse(BaseModel):
    """Response body for ``POST /generate``."""

    model: str
    prompt: str
    generated_text: str


class ModelInfo(BaseModel):
    """Single entry returned from ``GET /models``."""

    name: str
    description: str
    loaded: bool


class ModelsResponse(BaseModel):
    """Response body for ``GET /models``."""

    models: list[ModelInfo]


class ProjectionResponse(BaseModel):
    """Response body for ``GET /embeddings/projection``."""

    tokens: list[str]
    projection: list[list[float]]  # shape: (N, n_components)
    cluster_layers: list[list[int]]  # finest-grained layer first; -1 marks noise
    membership_layers: list[list[float]]  # per-token cluster confidence 0-1, parallel to layers


class NeighborsRequest(BaseModel):
    """Request body for ``POST /embeddings/neighbors``."""

    model: str
    query: str = Field(..., min_length=1, max_length=200)
    k: int = Field(8, ge=1, le=50)


class NeighborInfo(BaseModel):
    """A single neighbor entry — the matched token and its cosine similarity."""

    token: str
    similarity: float


class NeighborsResponse(BaseModel):
    """Response body for ``POST /embeddings/neighbors``."""

    tokenized_pieces: list[str]
    neighbors_per_piece: dict[str, list[NeighborInfo]]
