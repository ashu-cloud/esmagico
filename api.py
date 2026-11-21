"""FastAPI service exposing the invoice extraction pipeline (primary interface).

Run locally:
    uvicorn api:app --reload
Interactive docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from invoice_extractor.config import (
    REPO_ROOT,
    get_gemini_model,
    get_max_upload_size_bytes,
    get_max_upload_size_mb,
)
from invoice_extractor.extractor import (
    ProviderError,
    ProviderTimeoutError,
    extract_invoice,
)
from invoice_extractor.loader import SUPPORTED_EXTENSIONS, InvalidDocumentError

UPLOAD_CHUNK_BYTES = 64 * 1024  # 64 KB chunks to enforce size limit without buffering huge files.
TEMP_FILE_PREFIX = "invoice_upload_"
FRONTEND_INDEX = REPO_ROOT / "static" / "index.html"

app = FastAPI(
    title="Invoice Extraction API",
    description="Upload a PDF, JPG, or PNG document to extract structured invoice JSON.",
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the minimal document-review frontend wired to POST /extract."""
    return FileResponse(FRONTEND_INDEX)


def _validate_filename_and_extension(filename: str | None) -> tuple[str, str]:
    """Ensure the upload has a filename with an allowed extension (.pdf, .jpg, .png)."""
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or '(none)'}'. Allowed types: {allowed}.",
        )
    return safe_name, suffix


def _write_upload_to_temp_file(
    upload: UploadFile,
    suffix: str,
    max_bytes: int,
    max_mb: float,
) -> Path:
    """Stream upload into a temp file, rejecting empty or oversized uploads."""
    tmp = tempfile.NamedTemporaryFile(
        prefix=TEMP_FILE_PREFIX,
        suffix=suffix,
        delete=False,
    )
    tmp_path = Path(tmp.name)
    total_bytes = 0
    try:
        with tmp:
            while True:
                chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File exceeds maximum allowed size of {max_mb:g} MB.",
                    )
                tmp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


@app.get("/health")
def health() -> dict[str, str]:
    """Simple readiness probe (never exposes secrets)."""
    return {
        "status": "ok",
        "model": get_gemini_model(),
    }


@app.post("/extract")
def extract(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a single PDF/JPG/PNG upload and return extracted invoice JSON."""
    safe_name, suffix = _validate_filename_and_extension(file.filename)
    max_mb = get_max_upload_size_mb()
    max_bytes = get_max_upload_size_bytes()

    tmp_path: Path | None = None
    try:
        tmp_path = _write_upload_to_temp_file(
            upload=file,
            suffix=suffix,
            max_bytes=max_bytes,
            max_mb=max_mb,
        )
        result = extract_invoice(tmp_path, document_name=safe_name)
        return result.to_dict()
    except HTTPException:
        raise
    except (InvalidDocumentError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail="Model returned output that failed schema validation.",
        ) from exc
    except Exception as exc:
        # Never expose raw internal details or server secrets in HTTP responses.
        raise HTTPException(
            status_code=502,
            detail="Extraction failed due to an unexpected upstream or server error.",
        ) from exc
    finally:
        file.file.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
