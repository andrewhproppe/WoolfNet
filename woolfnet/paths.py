from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_LOCAL_URI = f"file:{DATA_DIR}/mlruns"
