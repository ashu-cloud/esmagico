"""Post-extraction schema and arithmetic validation (FR2, D5, D6)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from invoice_extractor.models import InvoiceExtraction

# Decision D6: absolute tolerance in the document's own currency.
ARITHMETIC_TOLERANCE = 0.05

REVIEW_METADATA_KEYS = ("needs_review", "review_reasons")


@dataclass
class ValidationOutcome:
    """Result of schema + arithmetic validation for one model output."""

    is_valid: bool
    model: InvoiceExtraction | None
    raw_data: dict[str, Any] | None
    reasons: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Final extraction output pairing schema fields with review metadata.

    Per 01_SPEC.md Section 5, `needs_review` and `review_reasons` are kept
    separate from the schema-checked invoice fields; removing those two keys
    yields an object that validates directly against schema.json.
    """

    data: dict[str, Any]
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return invoice fields plus review metadata for API/CLI/batch output."""
        payload = dict(self.data)
        payload["needs_review"] = self.needs_review
        payload["review_reasons"] = list(self.review_reasons)
        return payload

    def invoice_only(self) -> dict[str, Any]:
        """Return only the schema.json fields (review metadata stripped)."""
        return strip_review_metadata(self.to_dict())


def strip_review_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove needs_review and review_reasons so the dict validates against schema.json."""
    return {k: v for k, v in payload.items() if k not in REVIEW_METADATA_KEYS}


def _find_string_number_errors(raw: dict[str, Any]) -> list[str]:
    """Reject numeric fields returned as strings or booleans before Pydantic coerces them."""
    errors: list[str] = []
    for field_name in ("tax_total", "total"):
        val = raw.get(field_name)
        if isinstance(val, (str, bool)):
            errors.append(
                f"Schema error: '{field_name}' must be a plain JSON number or null, got {type(val).__name__}."
            )

    items = raw.get("line_items")
    if isinstance(items, list):
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            qty = item.get("quantity")
            if isinstance(qty, (str, bool)):
                errors.append(
                    f"Schema error: 'line_items[{idx}].quantity' must be a number or null, got {type(qty).__name__}."
                )
            amt = item.get("amount")
            if isinstance(amt, (str, bool)):
                errors.append(
                    f"Schema error: 'line_items[{idx}].amount' must be a plain JSON number, got {type(amt).__name__}."
                )
    return errors


def validate_extraction(
    raw_output: str | dict[str, Any],
    *,
    tolerance: float = ARITHMETIC_TOLERANCE,
) -> ValidationOutcome:
    """Run strict schema validation followed by the D6 arithmetic check."""
    if isinstance(raw_output, str):
        try:
            raw_dict = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            return ValidationOutcome(
                is_valid=False,
                model=None,
                raw_data=None,
                reasons=[f"Schema error: invalid JSON ({exc.msg})."],
            )
    elif isinstance(raw_output, dict):
        raw_dict = strip_review_metadata(raw_output)
    else:
        return ValidationOutcome(
            is_valid=False,
            model=None,
            raw_data=None,
            reasons=[f"Schema error: expected JSON object, got {type(raw_output).__name__}."],
        )

    if not isinstance(raw_dict, dict):
        return ValidationOutcome(
            is_valid=False,
            model=None,
            raw_data=None,
            reasons=["Schema error: top-level JSON value must be an object."],
        )

    reasons: list[str] = []

    # 1. Reject string-encoded numbers explicitly (04_EVAL_RULES.md).
    reasons.extend(_find_string_number_errors(raw_dict))

    # 2. Local Pydantic schema validation (types, patterns, extra='forbid', non-invoice null rule).
    model: InvoiceExtraction | None = None
    try:
        model = InvoiceExtraction.model_validate(raw_dict)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", ())) or "root"
            msg = err.get("msg", "validation error")
            reasons.append(f"Schema error at '{loc}': {msg}")

    if reasons or model is None:
        return ValidationOutcome(
            is_valid=False,
            model=model,
            raw_data=raw_dict,
            reasons=reasons,
        )

    # 3. Arithmetic check (only applies when is_invoice is true).
    if model.is_invoice:
        if model.line_items is None:
            reasons.append("Arithmetic check failed: 'line_items' is null on an invoice.")
        if model.tax_total is None:
            reasons.append(
                "Arithmetic check failed: 'tax_total' is null on an invoice (use 0 if no tax)."
            )
        if model.total is None:
            reasons.append("Arithmetic check failed: 'total' is null on an invoice.")

        if (
            model.line_items is not None
            and model.tax_total is not None
            and model.total is not None
        ):
            items_sum = sum(item.amount for item in model.line_items)
            expected_total = items_sum + model.tax_total
            diff = abs(expected_total - model.total)
            if diff > tolerance:
                reasons.append(
                    f"Arithmetic check failed: sum(line_items.amount) ({items_sum:.2f}) "
                    f"+ tax_total ({model.tax_total:.2f}) = {expected_total:.2f}, "
                    f"which differs from total ({model.total:.2f}) by {diff:.2f} "
                    f"(allowed tolerance: {tolerance:.2f})."
                )

    return ValidationOutcome(
        is_valid=len(reasons) == 0,
        model=model,
        raw_data=model.model_dump(),
        reasons=reasons,
    )
