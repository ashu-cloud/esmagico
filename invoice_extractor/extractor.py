"""Reusable vision LLM invoice extractor (independent of FastAPI and CLI).

Handles:
- Page-image loading (PDF pages -> PNG, JPG/PNG -> EXIF-corrected image)
- Gemini structured output with configurable timeout & exponential backoff (D7)
- Schema + arithmetic validation, single error-feedback retry, and review flagging (D5, D6)
- Token logging for every API call including retries (FR6)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from invoice_extractor.config import (
    get_api_max_retries,
    get_api_timeout_seconds,
    get_gemini_api_key,
    get_gemini_model,
)
from invoice_extractor.loader import PageImage, load_document_images
from invoice_extractor.models import InvoiceExtraction
from invoice_extractor.prompt import SYSTEM_PROMPT, USER_PROMPT, build_retry_prompt
from invoice_extractor.token_logger import log_token_usage
from invoice_extractor.validator import (
    ARITHMETIC_TOLERANCE,
    ExtractionResult,
    validate_extraction,
)

# Transient HTTP status codes worth retrying with exponential backoff (D7).
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
DEFAULT_BACKOFF_BASE_SECONDS = 1.0


class ProviderError(RuntimeError):
    """Raised when the upstream Gemini API fails or returns an invalid payload."""


class ProviderTimeoutError(ProviderError):
    """Raised when the upstream Gemini API times out."""


@dataclass(frozen=True)
class ModelCallResult:
    """Raw text and token metadata returned by one upstream model call."""

    text: str
    input_tokens: int
    output_tokens: int
    thoughts_tokens: int
    total_tokens: int
    latency_ms: int


# Optional hook signature used in tests to simulate broken outputs or API failures.
ModelCaller = Callable[[list[PageImage], str, str, float], ModelCallResult]


def _build_content_parts(pages: list[PageImage], prompt_text: str) -> list[types.Part]:
    """Attach all document page images followed by the text instruction."""
    parts: list[types.Part] = [
        types.Part.from_bytes(data=page.data, mime_type=page.mime_type)
        for page in pages
    ]
    parts.append(types.Part.from_text(text=prompt_text))
    return parts


def _extract_token_counts(
    response: types.GenerateContentResponse,
) -> tuple[int, int, int, int]:
    """Read input, billable output (candidates + thoughts), thoughts, and total tokens."""
    usage = response.usage_metadata
    if usage is None:
        return 0, 0, 0, 0
    input_tokens = int(usage.prompt_token_count or 0)
    candidates_tokens = int(usage.candidates_token_count or 0)
    thoughts_tokens = int(usage.thoughts_token_count or 0)
    # Gemini bills thinking tokens at the output-token rate.
    output_tokens = candidates_tokens + thoughts_tokens
    total_tokens = int(usage.total_token_count or (input_tokens + output_tokens))
    return input_tokens, output_tokens, thoughts_tokens, total_tokens


def _is_timeout_exception(exc: Exception) -> bool:
    """Detect timeout exceptions across stdlib, httpx/httpcore, and Gemini SDK."""
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return True
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "")
        return code == 504 or "DEADLINE_EXCEEDED" in status
    return False


def _is_transient_error(exc: Exception) -> bool:
    """Return True if an upstream failure is transient and safe to retry with backoff."""
    if _is_timeout_exception(exc):
        return True
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        return code in TRANSIENT_HTTP_CODES
    name = type(exc).__name__.lower()
    return any(tok in name for tok in ("connect", "network", "transport", "remoteprotocol"))


def _default_gemini_caller(
    pages: list[PageImage],
    prompt_text: str,
    model: str,
    timeout_seconds: float,
) -> ModelCallResult:
    """Execute one Gemini generate_content call with an explicit HTTP timeout."""
    client = genai.Client(
        api_key=get_gemini_api_key(),
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )
    # Use response_json_schema (standard JSON Schema) because Gemini's legacy
    # response_schema endpoint rejects Pydantic's additionalProperties: false.
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        response_mime_type="application/json",
        response_json_schema=InvoiceExtraction.model_json_schema(),
    )

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=_build_content_parts(pages, prompt_text),
        config=config,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    input_tokens, output_tokens, thoughts_tokens, total_tokens = _extract_token_counts(
        response
    )
    if not response.text:
        raise ProviderError("Vision provider returned an empty response.")

    return ModelCallResult(
        text=response.text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thoughts_tokens=thoughts_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )


def _call_with_backoff(
    *,
    caller: ModelCaller,
    pages: list[PageImage],
    prompt_text: str,
    model: str,
    document_name: str,
    validation_attempt: int,
    timeout_seconds: float,
    max_api_retries: int,
    backoff_base_seconds: float,
    log_file: str | Path | None,
) -> ModelCallResult:
    """Call the upstream model with timeout and bounded exponential backoff (D7)."""
    last_exc: Exception | None = None

    for transport_try in range(max_api_retries + 1):
        start = time.perf_counter()
        try:
            call_res = caller(pages, prompt_text, model, timeout_seconds)
            log_token_usage(
                document=document_name,
                model=model,
                input_tokens=call_res.input_tokens,
                output_tokens=call_res.output_tokens,
                thoughts_tokens=call_res.thoughts_tokens,
                total_tokens=call_res.total_tokens,
                attempt=validation_attempt,
                latency_ms=call_res.latency_ms,
                status="ok",
                log_file=log_file,
            )
            return call_res
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            last_exc = exc
            safe_code = getattr(exc, "code", None)
            safe_status = getattr(exc, "status", None) or type(exc).__name__
            log_token_usage(
                document=document_name,
                model=model,
                input_tokens=0,
                output_tokens=0,
                attempt=validation_attempt,
                latency_ms=latency_ms,
                status="error",
                error=f"{safe_status}({safe_code})" if safe_code is not None else str(safe_status),
                log_file=log_file,
            )

            can_retry = _is_transient_error(exc) and (transport_try < max_api_retries)
            if can_retry:
                sleep_s = backoff_base_seconds * (2**transport_try)
                time.sleep(sleep_s)
                continue
            break

    assert last_exc is not None
    if _is_timeout_exception(last_exc):
        raise ProviderTimeoutError(
            f"Vision provider timed out after {max_api_retries + 1} attempt(s)."
        ) from last_exc
    if isinstance(last_exc, ProviderError):
        raise last_exc
    safe_code = getattr(last_exc, "code", None)
    if safe_code is not None:
        raise ProviderError(
            f"Vision provider request failed with HTTP {safe_code}."
        ) from last_exc
    raise ProviderError(
        f"Vision provider request failed ({type(last_exc).__name__})."
    ) from last_exc


def _fallback_payload(raw_data: dict | None, raw_text: str) -> dict:
    """Build a safe JSON dict if schema validation still failed on retry."""
    if raw_data is not None:
        return raw_data
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {
        "is_invoice": None,
        "vendor": None,
        "invoice_number": None,
        "invoice_date": None,
        "currency": None,
        "line_items": None,
        "tax_total": None,
        "total": None,
    }


def extract_invoice(
    file_path: str | Path,
    *,
    model: str | None = None,
    log_file: str | Path | None = None,
    document_name: str | None = None,
    tolerance: float = ARITHMETIC_TOLERANCE,
    timeout_seconds: float | None = None,
    max_api_retries: int | None = None,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    call_model_fn: ModelCaller | None = None,
) -> ExtractionResult:
    """Extract, validate, retry once on failure, and flag if still invalid.

    Returns an ExtractionResult whose .to_dict() contains the 8 schema.json
    fields plus `needs_review` and `review_reasons`.
    """
    path = Path(file_path)
    logged_name = document_name or path.name
    chosen_model = get_gemini_model(model)
    req_timeout = (
        timeout_seconds if timeout_seconds is not None else get_api_timeout_seconds()
    )
    api_retries = (
        max_api_retries if max_api_retries is not None else get_api_max_retries()
    )
    caller = call_model_fn or _default_gemini_caller

    pages = load_document_images(path)

    # Attempt 1: initial extraction.
    first_call = _call_with_backoff(
        caller=caller,
        pages=pages,
        prompt_text=USER_PROMPT,
        model=chosen_model,
        document_name=logged_name,
        validation_attempt=1,
        timeout_seconds=req_timeout,
        max_api_retries=api_retries,
        backoff_base_seconds=backoff_base_seconds,
        log_file=log_file,
    )
    first_check = validate_extraction(first_call.text, tolerance=tolerance)
    if first_check.is_valid and first_check.model is not None:
        return ExtractionResult(
            data=first_check.model.model_dump(),
            needs_review=False,
            review_reasons=[],
        )

    # Attempt 2 (D5): retry ONCE feeding back the specific validation failures.
    retry_prompt = build_retry_prompt(first_call.text, first_check.reasons)
    second_call = _call_with_backoff(
        caller=caller,
        pages=pages,
        prompt_text=retry_prompt,
        model=chosen_model,
        document_name=logged_name,
        validation_attempt=2,
        timeout_seconds=req_timeout,
        max_api_retries=api_retries,
        backoff_base_seconds=backoff_base_seconds,
        log_file=log_file,
    )
    second_check = validate_extraction(second_call.text, tolerance=tolerance)
    if second_check.is_valid and second_check.model is not None:
        return ExtractionResult(
            data=second_check.model.model_dump(),
            needs_review=False,
            review_reasons=[],
        )

    # Still failing after retry: never silently pass bad data; flag for human review.
    payload = _fallback_payload(second_check.raw_data, second_call.text)
    return ExtractionResult(
        data=payload,
        needs_review=True,
        review_reasons=second_check.reasons,
    )
