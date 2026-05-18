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
