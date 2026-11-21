"""Append-only JSONL logger for model token usage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invoice_extractor.config import LOGS_DIR

DEFAULT_TOKEN_LOG = LOGS_DIR / "token_usage.jsonl"


def log_token_usage(
    *,
    document: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    thoughts_tokens: int = 0,
    total_tokens: int | None = None,
    attempt: int = 1,
    latency_ms: int | None = None,
    status: str = "ok",
    error: str | None = None,
    log_file: str | Path | None = None,
) -> Path:
    """Append one token-usage record to the JSONL log file.

    Note: output_tokens includes thinking tokens when present because Gemini
    bills thinking tokens at the output-token rate.
    """
    target = Path(log_file) if log_file is not None else DEFAULT_TOKEN_LOG
    target.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "document": document,
        "model": model,
        "attempt": attempt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "total_tokens": (
            total_tokens if total_tokens is not None else input_tokens + output_tokens
        ),
        "latency_ms": latency_ms,
        "status": status,
    }
    if error is not None:
        record["error"] = error

    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return target
