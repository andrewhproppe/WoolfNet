from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_LOCAL_URI = f"file:{DATA_DIR}/mlruns"
MLFLOW_LOCAL_ARTIFACTS = DATA_DIR / "mlflow_artifacts"
MLFLOW_LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
