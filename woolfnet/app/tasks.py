"""CLI entrypoint for the FastAPI server (``woolf serve``)."""

import logging

import click
import uvicorn

logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Enable uvicorn auto-reload (dev only).")
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI inference server."""
    uvicorn.run("woolfnet.api.app:app", host=host, port=port, reload=reload)


COMMANDS = {
    "serve": serve,
}
