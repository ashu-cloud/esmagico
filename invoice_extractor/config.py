"""Configuration helpers for paths and environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root is one level above this package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = REPO_ROOT / "logs"

# Load .env once when config is imported so CLI entrypoints see local settings.
load_dotenv(REPO_ROOT / ".env")

DEFAULT_SAMPLE_PACK_DIR = "Sample Pack"
DEFAULT_MAX_UPLOAD_SIZE_MB = 20
DEFAULT_API_TIMEOUT_SECONDS = 45.0
DEFAULT_API_MAX_RETRIES = 2
DEFAULT_API_BACKOFF_BASE_SECONDS = 1.0
_PLACEHOLDER_API_KEY = "your_gemini_api_key_here"


def get_sample_pack_dir(cli_override: str | Path | None = None) -> Path:
    """Resolve the sample pack path from CLI override, env var, or default."""
    raw_path = (
        str(cli_override)
        if cli_override is not None
        else os.getenv("SAMPLE_PACK_DIR", DEFAULT_SAMPLE_PACK_DIR)
    )
    path = Path(raw_path).expanduser()
    # Treat relative paths as relative to repo root so scripts work from any CWD.
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def get_labelled_dir(cli_override: str | Path | None = None) -> Path:
    """Return the path to the labelled documents folder inside the sample pack."""
    return get_sample_pack_dir(cli_override) / "labelled"


def get_unlabelled_dir(cli_override: str | Path | None = None) -> Path:
    """Return the path to the unlabelled documents folder inside the sample pack."""
    return get_sample_pack_dir(cli_override) / "unlabelled"


def get_schema_path(cli_override: str | Path | None = None) -> Path:
    """Return the path to schema.json inside the sample pack."""
    return get_sample_pack_dir(cli_override) / "schema.json"


def get_gemini_api_key() -> str:
    """Return GEMINI_API_KEY from the environment or raise a clear error."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == _PLACEHOLDER_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def get_gemini_model(cli_override: str | None = None) -> str:
    """Return the configured Gemini model ID."""
    if cli_override:
        return cli_override.strip()
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()


def get_max_upload_size_mb() -> float:
    """Return the configured maximum upload size in megabytes."""
    raw = os.getenv("MAX_UPLOAD_SIZE_MB", str(DEFAULT_MAX_UPLOAD_SIZE_MB)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"MAX_UPLOAD_SIZE_MB must be a number, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError("MAX_UPLOAD_SIZE_MB must be greater than 0.")
    return value


def get_max_upload_size_bytes() -> int:
    """Return the configured maximum upload size in bytes."""
    return int(get_max_upload_size_mb() * 1024 * 1024)


def get_api_timeout_seconds() -> float:
    """Return the upstream Gemini API request timeout in seconds."""
    raw = os.getenv("API_TIMEOUT_SECONDS", str(DEFAULT_API_TIMEOUT_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"API_TIMEOUT_SECONDS must be a number, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError("API_TIMEOUT_SECONDS must be greater than 0.")
    return value


def get_api_max_retries() -> int:
    """Return max transient API retries with exponential backoff."""
    raw = os.getenv("API_MAX_RETRIES", str(DEFAULT_API_MAX_RETRIES)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"API_MAX_RETRIES must be an integer, got: {raw!r}") from exc
    if value < 0:
        raise ValueError("API_MAX_RETRIES cannot be negative.")
    return value


def get_api_backoff_base_seconds() -> float:
    """Return base delay (seconds) for exponential backoff on transient API errors."""
    raw = os.getenv(
        "API_BACKOFF_BASE_SECONDS", str(DEFAULT_API_BACKOFF_BASE_SECONDS)
    ).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"API_BACKOFF_BASE_SECONDS must be a number, got: {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError("API_BACKOFF_BASE_SECONDS cannot be negative.")
    return value
