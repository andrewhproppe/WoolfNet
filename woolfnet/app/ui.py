"""Gradio frontend for WoolfNet, calling the FastAPI backend over HTTP."""

import logging
import os

import gradio as gr
import numpy as np
import requests

from woolfnet.app.visualization import build_embedding_plot

logger = logging.getLogger(__name__)

API_URL = os.environ.get("WOOLFNET_API_URL", "http://localhost:8000")
DEFAULT_PROMPT = "Mrs Dalloway said she would buy the flowers herself."
DEFAULT_QUERY = "sea"


def _http_error_message(exc: requests.RequestException) -> str:
    """Render an HTTP exception as a single user-facing string."""
    if isinstance(exc, requests.HTTPError):
        return f"API error: {exc.response.status_code}. {exc.response.text}"
    return f"Could not reach API at {API_URL}: {exc}"


def _fetch_models() -> list[str]:
    """List available model names from the API, falling back to a hardcoded set if unreachable."""
    try:
        r = requests.get(f"{API_URL}/models", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json()["models"]]
    except Exception as e:
        logger.warning(f"Could not fetch /models from {API_URL}: {e}")
        return ["woolf-scratch", "gpt2", "gpt2-woolf", "gpt2-woolf-deep"]


def _generate(prompt: str, model: str, max_new_tokens: int, temperature: float) -> str:
    """Hit POST /generate and return the continuation, or an error string."""
    if not prompt.strip():
        return "(empty prompt)"
    try:
        r = requests.post(
            f"{API_URL}/generate",
            json={
                "prompt": prompt,
                "model": model,
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["generated_text"]
    except requests.RequestException as e:
        return _http_error_message(e)


def _format_neighbor_panel(neighbors_payload: dict) -> str:
    pieces = neighbors_payload["tokenized_pieces"]
    per_piece = neighbors_payload["neighbors_per_piece"]
    lines = [f"**Your query tokenised into:** `{' · '.join(pieces)}`", ""]
    for piece in pieces:
        nbrs = per_piece.get(piece, [])
        if not nbrs:
            lines.append(f"### `{piece}`")
            lines.append("_not present in the top-frequency vocabulary subset_")
            lines.append("")
            continue
        lines.append(f"### `{piece}`, top {len(nbrs)} neighbors")
        for n in nbrs:
            lines.append(f"- `{n['token']}` &nbsp;&nbsp; *{n['similarity']:.3f}*")
        lines.append("")
    return "\n".join(lines)


def _load_projection(model: str, dim_label: str):
    dim = 3 if dim_label.startswith("3") else 2
    try:
        r = requests.get(
            f"{API_URL}/embeddings/projection",
            params={"model": model, "dim": dim},
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        gr.Warning(_http_error_message(e))
        return None, None, gr.update()

    cluster_layers = [np.asarray(layer, dtype=np.int32) for layer in data["cluster_layers"]]
    membership_layers = [np.asarray(m, dtype=np.float32) for m in data["membership_layers"]]
    coarsest_idx = len(cluster_layers) - 1
    state = {
        "model": model,
        "dim": dim,
        "tokens": data["tokens"],
        "projection": np.asarray(data["projection"], dtype=np.float32),
        "cluster_layers": cluster_layers,
        "membership_layers": membership_layers,
        "layer_idx": coarsest_idx,
        "shade_by_confidence": False,
    }
    fig = build_embedding_plot(
        state["projection"],
        state["tokens"],
        cluster_ids=cluster_layers[coarsest_idx],
        title=_plot_title(model, dim, coarsest_idx, cluster_layers),
    )
    slider_update = gr.update(
        minimum=0,
        maximum=coarsest_idx,
        value=coarsest_idx,
        step=1,
        visible=coarsest_idx > 0,
    )
    return state, fig, slider_update


def _plot_title(model: str, dim: int, layer_idx: int, layers: list[np.ndarray]) -> str:
    n_clusters = len(set(layers[layer_idx].tolist()) - {-1})
    return f"{model}, {dim}D, EVoC layer {layer_idx} ({n_clusters} clusters)"


def _redraw(state: dict | None) -> "object":
    """Rebuild the plot from current state (layer + shading toggle). Returns a Figure."""
    if state is None:
        return gr.update()
    layers = state["cluster_layers"]
    idx = state["layer_idx"]
    memberships = state["membership_layers"][idx] if state.get("shade_by_confidence") else None
    return build_embedding_plot(
        state["projection"],
        state["tokens"],
        cluster_ids=layers[idx],
        membership_strengths=memberships,
        title=_plot_title(state["model"], state["dim"], idx, layers),
    )


def _on_granularity_change(state: dict | None, layer_idx: float):
    if state is None:
        return gr.update(), state
    layers = state["cluster_layers"]
    state["layer_idx"] = max(0, min(int(layer_idx), len(layers) - 1))
    return _redraw(state), state


def _on_shade_toggle(state: dict | None, shade: bool):
    if state is None:
        return gr.update(), state
    state["shade_by_confidence"] = bool(shade)
    return _redraw(state), state


def _run_query(state: dict | None, model: str, query: str, top_k: int):
    """Look up neighbors for the query, returning an updated plot and the textual panel.

    Returns ``gr.update()`` for the plot when the state doesn't match the current model,
    so the previous plot stays visible while we surface a warning instead.
    """
    if state is None or state.get("model") != model:
        gr.Warning("Projection not loaded yet for this model. Click 'Load embeddings' first.")
        return gr.update(), "_(no embeddings loaded)_"
    if not query.strip():
        return gr.update(), "_(empty query)_"

    try:
        r = requests.post(
            f"{API_URL}/embeddings/neighbors",
            json={"model": model, "query": query, "k": int(top_k)},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
    except requests.RequestException as e:
        return gr.update(), _http_error_message(e)

    pieces = payload["tokenized_pieces"]
    neighbor_tokens = [
        n["token"] for piece in pieces for n in payload["neighbors_per_piece"].get(piece, [])
    ]
    layers = state["cluster_layers"]
    idx = state.get("layer_idx", len(layers) - 1)
    memberships = state["membership_layers"][idx] if state.get("shade_by_confidence") else None
    fig = build_embedding_plot(
        state["projection"],
        state["tokens"],
        cluster_ids=layers[idx],
        membership_strengths=memberships,
        query_pieces=pieces,
        neighbor_tokens=neighbor_tokens,
        title=f"{model}, {state['dim']}D, query: '{query}'",
    )
    return fig, _format_neighbor_panel(payload)


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks app with Generate and Explore-embeddings tabs."""
    models = _fetch_models()
    default_model = "gpt2-woolf" if "gpt2-woolf" in models else models[0]

    with gr.Blocks(title="WoolfNet") as demo:
        gr.Markdown(
            "# WoolfNet\n"
            "Generate text in the style of Virginia Woolf, and explore how each model "
            "organises its token-embedding space."
        )
        with gr.Tabs():
            with gr.Tab("Generate"):
                _build_generate_tab(models, default_model)
            with gr.Tab("Explore embeddings"):
                _build_embeddings_tab(models, default_model)

    return demo


def _build_generate_tab(models: list[str], default_model: str) -> None:
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
            with gr.Row():
                model = gr.Dropdown(choices=models, value=default_model, label="Model")
                max_tokens = gr.Slider(8, 256, value=80, step=8, label="Max new tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            output = gr.Textbox(label="Generated text", lines=14)

    run.click(_generate, inputs=[prompt, model, max_tokens, temperature], outputs=output)


def _build_embeddings_tab(models: list[str], default_model: str) -> None:
    state = gr.State(value=None)

    gr.Markdown(
        "_Load the embeddings for a model, then enter a word to see its position "
        "and nearest neighbors in the projected space. The query is tokenised with the "
        "model's own BPE tokenizer, so short words may split into multiple pieces._"
    )
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Dropdown(choices=models, value=default_model, label="Model")
            dim = gr.Radio(choices=["2D", "3D"], value="2D", label="Projection")
            load_btn = gr.Button("Load embeddings", variant="secondary")
            granularity = gr.Slider(
                minimum=0,
                maximum=1,
                value=0,
                step=1,
                label="Cluster granularity (fine → coarse)",
                visible=False,
            )
            shade_confidence = gr.Checkbox(
                value=False,
                label="Shade points by EVoC membership confidence",
            )
            query = gr.Textbox(label="Query", value=DEFAULT_QUERY, placeholder="e.g. sea")
            top_k = gr.Slider(3, 25, value=8, step=1, label="Top-k neighbors")
            search_btn = gr.Button("Search", variant="primary")
            neighbor_panel = gr.Markdown("_(load embeddings, then search a word)_")
        with gr.Column(scale=3):
            plot = gr.Plot(label=None)

    search_inputs = [state, model, query, top_k]
    search_outputs = [plot, neighbor_panel]
    load_btn.click(_load_projection, inputs=[model, dim], outputs=[state, plot, granularity])
    granularity.release(_on_granularity_change, inputs=[state, granularity], outputs=[plot, state])
    shade_confidence.change(
        _on_shade_toggle, inputs=[state, shade_confidence], outputs=[plot, state]
    )
    search_btn.click(_run_query, inputs=search_inputs, outputs=search_outputs)
    query.submit(_run_query, inputs=search_inputs, outputs=search_outputs)


def serve(host: str = "0.0.0.0", port: int = 7860):
    """Launch the Gradio UI bound to host:port."""
    build_ui().launch(server_name=host, server_port=port)
