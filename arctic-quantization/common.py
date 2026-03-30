import os
from pathlib import Path


BASE_MODEL_ID = "Snowflake/Arctic-Text2SQL-R1-7B"
DEFAULT_GPTQ8_REPO_ID = "ByteMaster01/arctic-text2sql-r1-7b-gptq8"
DEFAULT_TENSORIZED_REPO_ID = "ByteMaster01/arctic-text2sql-r1-7b-gptq8-tensorized"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_GPTQ8_DIR = MODELS_DIR / "arctic-text2sql-r1-7b-gptq8"
DEFAULT_Q8_DIR = MODELS_DIR / "arctic-text2sql-r1-7b-q8"
DEFAULT_TENSORIZED_DIR = MODELS_DIR / "arctic-text2sql-r1-7b-gptq8-tensorized"


def load_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    for env_path in (Path(".env"), PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "HF_TOKEN":
                return value.strip().strip("'\"")

    return None


def ensure_hf_token() -> str:
    token = load_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment or .env.")
    return token
