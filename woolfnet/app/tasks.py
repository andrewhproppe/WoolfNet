"""CLI entrypoints for the FastAPI server (``woolf serve``) and Gradio UI (``woolf ui``)."""

import logging

import click
import uvicorn

from woolfnet.app.ui import serve as serve_ui

logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Enable uvicorn auto-reload (dev only).")
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI inference server."""
    uvicorn.run("woolfnet.api.app:app", host=host, port=port, reload=reload)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=7860, show_default=True, type=int)
def ui(host: str, port: int):
    """Launch the Gradio UI (expects the API at WOOLFNET_API_URL, default localhost:8000)."""
    serve_ui(host=host, port=port)


COMMANDS = {
    "serve": serve,
    "ui": ui,
}
