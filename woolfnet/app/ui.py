"""Gradio frontend that calls the WoolfNet FastAPI backend over HTTP."""

import logging
import os

import gradio as gr
import requests

logger = logging.getLogger(__name__)

API_URL = os.environ.get("WOOLFNET_API_URL", "http://localhost:8000")
DEFAULT_PROMPT = "Mrs Dalloway said she would buy the flowers herself."


def _fetch_models() -> list[str]:
    try:
        r = requests.get(f"{API_URL}/models", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json()["models"]]
    except Exception as e:
        logger.warning(f"Could not fetch /models from {API_URL}: {e}")
        return ["woolf-scratch", "gpt2", "gpt2-woolf"]


def _generate(prompt: str, model: str, max_new_tokens: int, temperature: float) -> str:
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
    except requests.HTTPError as e:
        return f"API error: {e.response.status_code} — {e.response.text}"
    except requests.RequestException as e:
        return f"Could not reach API at {API_URL}: {e}"


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    models = _fetch_models()
    default_model = "gpt2-woolf" if "gpt2-woolf" in models else models[0]

    with gr.Blocks(title="WoolfNet") as demo:
        gr.Markdown(
            "# WoolfNet\n"
            "Generate text in the style of Virginia Woolf. Choose between a small "
            "GPT trained from scratch, base GPT-2, or GPT-2 fine-tuned on Woolf's works."
        )
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

    return demo


def serve(host: str = "0.0.0.0", port: int = 7860):
    """Launch the Gradio UI bound to ``host:port``."""
    build_ui().launch(server_name=host, server_port=port)
