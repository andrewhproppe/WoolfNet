"""``WoolfModel``: a unified inference model loaded by name from ``configs/inference.yml``."""

import logging
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

from woolfnet.config import DotDict, load_yaml
from woolfnet.gpt.config import GPTConfig
from woolfnet.gpt.model import GPT
from woolfnet.gpt.utils import generate_text, generate_text_hf
from woolfnet.paths import ROOT_DIR
from woolfnet.utils.mlflow import load_mlflow_artifacts, resolve_run_id

logger = logging.getLogger(__name__)


class WoolfModel:
    """An inference model selected by name from ``configs/inference.yml``.

    The config entry's ``type`` field dispatches loading. ``torch`` loads our in-house
    PyTorch GPT + its BPE tokenizer; ``huggingface`` loads a ``GPT2LMHeadModel`` either
    from a local checkpoint dir or from the HuggingFace Hub.

    Artifacts can be sourced three ways, in order of preference:

    1. ``run_id``  — an MLflow run UUID. Artifacts are downloaded and cached locally.
    2. ``name``    — a versioned MLflow run name (resolved via ``params.name``).
    3. legacy path-style fields (``weights``/``model_config``/``tokenizer`` for torch;
       ``path`` for huggingface), kept for entries that predate MLflow logging.
    """

    CONFIG_PATH = ROOT_DIR / "configs" / "inference.yml"
    MLFLOW_EXPERIMENT = "gpt"

    def __init__(self, name: str, device: str | None = None):
        spec = self._spec(name)
        self.name = name
        self.source: str = spec.type
        self.description: str = spec.description
        self.device: str = device or self._auto_device()

        if self.source == "torch":
            self._load_torch(spec)
        elif self.source == "huggingface":
            self._load_huggingface(spec)
        elif self.source == "jax":
            raise NotImplementedError("JAX inference models not yet supported.")
        else:
            raise ValueError(f"Unknown model type '{self.source}' for '{name}'.")

        logger.info(f"Loaded {name} ({self.source}) on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 60, temperature: float = 0.8) -> str:
        """Return ``prompt`` continued by ``max_new_tokens`` newly sampled tokens."""
        if self.source == "torch":
            return generate_text(
                self.model, self.tokenizer, prompt, max_new_tokens, temperature, device=self.device
            )
        return generate_text_hf(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=self.device,
        )

    @classmethod
    def available(cls) -> list[str]:
        """Names whose underlying artifacts are reachable right now."""
        specs = cls._specs()
        return [n for n in specs.keys() if cls._is_available(getattr(specs, n))]

    @classmethod
    def descriptions(cls) -> dict[str, str]:
        """Map of name → description for every configured model (available or not)."""
        specs = cls._specs()
        return {n: getattr(specs, n).description for n in specs.keys()}

    def _load_torch(self, spec: DotDict) -> None:
        if self._is_mlflow_spec(spec):
            root = self._mlflow_artifact_dir(spec)
            weights = root / "model.pt"
            model_config_path = root / "gpt_config.yml"
            tokenizer_dir = root / "tokenizer"
        else:
            weights = self._resolve(spec.weights)
            model_config_path = self._resolve(spec.model_config)
            tokenizer_dir = self._resolve(spec.tokenizer)

        if not weights.exists():
            raise FileNotFoundError(f"No weights file at {weights}")

        config = GPTConfig.from_yaml(model_config_path)
        self.model = GPT(config).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model.eval()
        self.tokenizer = ByteLevelBPETokenizer(
            str(tokenizer_dir / "vocab.json"),
            str(tokenizer_dir / "merges.txt"),
        )

    def _load_huggingface(self, spec: DotDict) -> None:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        if self._is_mlflow_spec(spec):
            path = str(self._mlflow_artifact_dir(spec))
        else:
            resolved = self._resolve(spec.path)
            path = str(resolved) if resolved.exists() else spec.path

        self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def _mlflow_artifact_dir(cls, spec: DotDict) -> Path:
        """Resolve a spec's ``run_id`` or ``name`` to a local artifact directory."""
        value = getattr(spec, "run_id", None) or getattr(spec, "name", None)
        run_id = resolve_run_id(value, experiment_name=cls.MLFLOW_EXPERIMENT)
        return load_mlflow_artifacts(run_id)

    @classmethod
    def _specs(cls) -> DotDict:
        return load_yaml(cls.CONFIG_PATH).models

    @classmethod
    def _spec(cls, name: str) -> DotDict:
        specs = cls._specs()
        if name not in specs.keys():
            raise ValueError(f"Unknown model '{name}'. Configured: {list(specs.keys())}")
        return getattr(specs, name)

    @staticmethod
    def _is_mlflow_spec(spec: DotDict) -> bool:
        return hasattr(spec, "run_id") or hasattr(spec, "name")

    @staticmethod
    def _is_available(spec: DotDict) -> bool:
        # MLflow-sourced entries are assumed available; failures surface at load time
        # rather than blocking the /models endpoint on a remote round-trip.
        if WoolfModel._is_mlflow_spec(spec):
            return True
        if spec.type == "torch":
            return (
                WoolfModel._resolve(spec.weights).exists()
                and WoolfModel._resolve(spec.tokenizer).exists()
            )
        if spec.type == "huggingface":
            # A path with no separator is treated as a HuggingFace Hub id (always reachable).
            return "/" not in spec.path or WoolfModel._resolve(spec.path).exists()
        return False

    @staticmethod
    def _resolve(path: str) -> Path:
        p = Path(path)
        if p.is_absolute() or p.exists():
            return p
        return ROOT_DIR / path

    @staticmethod
    def _auto_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
