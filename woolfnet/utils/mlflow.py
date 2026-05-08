import logging
import re
import tempfile

import mlflow

from woolfnet.config import DotDict

logger = logging.getLogger(__name__)


def resolve_run_id(value: str, experiment_name: str | None = None) -> str:
    """Resolve a 32-char hex run ID or model name to an MLflow run ID."""
    if re.fullmatch(r"[0-9a-f]{32}", value):
        return value

    kwargs: dict = {
        "filter_string": f'params.name = "{value}"',
        "max_results": 10,
    }
    if experiment_name:
        kwargs["experiment_names"] = [experiment_name]

    runs = mlflow.search_runs(**kwargs)
    if runs.empty:
        raise ValueError(f"No MLflow run found with params.name = '{value}'.")
    if len(runs) > 1:
        details = "\n".join(
            f"  {row['run_id']}  (started {row['start_time']})" for _, row in runs.iterrows()
        )
        raise ValueError(
            f"Multiple runs match params.name = '{value}':\n{details}\n"
            "Pass the exact run ID to disambiguate."
        )
    return runs.iloc[0]["run_id"]


def _parse_version(version_str: str) -> tuple[int, ...] | None:
    """Parse a version string (e.g. ``0.0.1``, ``v0``) into an int tuple, or None."""
    try:
        return tuple(int(p) for p in version_str.lstrip("v").split("."))
    except ValueError:
        return None


def next_model_version(
    name: str, base_version: str = "0.0", experiment_name: str | None = None
) -> str:
    """Return the next patch-bumped version for a model name within a ``major.minor`` base.

    Searches MLflow for existing runs matching ``{name}_{base_version}.*`` and increments
    the patch number. Returns ``"{base_version}.1"`` when no prior runs exist.
    """
    base_parsed = _parse_version(base_version)
    if base_parsed is None or len(base_parsed) < 2:
        raise ValueError(f"base_version must be a 'major.minor' string, got '{base_version}'.")
    major, minor = base_parsed[0], base_parsed[1]

    kwargs: dict = {"filter_string": "params.name != ''", "max_results": 1000}
    if experiment_name:
        kwargs["experiment_names"] = [experiment_name]

    runs = mlflow.search_runs(**kwargs)
    if runs.empty:
        return f"{major}.{minor}.1"

    prefix = f"{name}_"
    max_patch = 0
    for param_name in runs["params.name"]:
        if not param_name.startswith(prefix):
            continue
        parsed = _parse_version(param_name[len(prefix) :])
        if parsed is None:
            continue
        normalized = (parsed + (0, 0, 0))[:3]
        if normalized[0] == major and normalized[1] == minor and normalized[2] > max_patch:
            max_patch = normalized[2]

    next_ver = f"{major}.{minor}.{max_patch + 1}"
    logger.info(f"Auto-versioning '{name}': {major}.{minor}.{max_patch} -> {next_ver}")
    return next_ver


def configure_mlflow(experiment_name: str, set_null: bool) -> tuple[str, str]:
    """Configure MLflow tracking URI and experiment name, with option to set as null"""

    # If set_null, still run in MLflow context, but do not persist any artifacts
    tracking_uri = mlflow.get_tracking_uri() if not set_null else f"file://{tempfile.mkdtemp()}"
    experiment_name = experiment_name if not set_null else "null"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri, experiment_name


def make_versioned_model_name(config: DotDict, set_null: bool) -> tuple[str, DotDict]:
    """
    Small helper function that updates a config version number, and returns the verisioned
    model name and config.
    """
    if not set_null:
        config.version = next_model_version(
            config.name, base_version=str(getattr(config, "version", "0.0"))
        )
    else:
        config.version = getattr(config, "version", "dev")
    name = f"{config.name}_{config.version}"

    return name, config
