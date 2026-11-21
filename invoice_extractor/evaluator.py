"""Field-by-field evaluation harness for labelled invoice documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

LEGAL_SUFFIXES = {
    "pvt",
    "ltd",
    "limited",
    "private",
    "llp",
    "llc",
    "inc",
    "gmbh",
    "ag",
    "bv",
    "co",
}

DESCRIPTION_SIMILARITY_THRESHOLD = 0.85
AMOUNT_TOLERANCE = 0.01
QUANTITY_TOLERANCE = 0.001

SCALAR_FIELDS = (
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "tax_total",
    "total",
)


@dataclass
class FieldError:
    """One mismatched field (or line-item row) for verbose reporting."""

    doc_id: str
    field: str
    expected: Any
    actual: Any
    note: str = ""


@dataclass
class LineItemStats:
    """Row-level and sub-field line item counts for one document or aggregate."""

    total_rows: int = 0
    gt_rows: int = 0
    pred_rows: int = 0
    correct_rows: int = 0
    desc_correct: int = 0
    qty_correct: int = 0
    amount_correct: int = 0


@dataclass
class DocumentEval:
    """Evaluation result for a single labelled document."""

    doc_id: str
    field_matches: dict[str, bool]
    all_fields_correct: bool
    line_item_stats: LineItemStats
    needs_review: bool
    review_reasons: list[str]
    string_number_failures: int
    errors: list[FieldError] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Aggregated metrics across all evaluated labelled documents."""

    total_docs: int
    field_correct: dict[str, int]
    all_fields_correct_docs: int
    line_item_stats: LineItemStats
    needs_review_docs: int
    flagged_wrong_docs: int
    unflagged_wrong_docs: int
    flagged_correct_docs: int
    string_number_failures: int
    documents: list[DocumentEval]


def normalize_vendor(value: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, and drop trailing legal suffixes."""
    text = value.lower().strip()
    # Keep "b.v." as a single token before stripping punctuation.
    text = re.sub(r"\bb\.v\.?", " bv ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_description(value: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = re.sub(r"[^\w\s]", " ", value.lower())
    return " ".join(text.split())


def parse_numeric(value: Any) -> tuple[float | None, bool]:
    """Parse a numeric value, flagging string-encoded numbers as schema failures.

    Returns (parsed_float_or_none, was_string_number).
    """
    if value is None or isinstance(value, bool):
        return None, False
    if isinstance(value, (int, float)):
        return float(value), False
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        try:
            return float(cleaned), True
        except ValueError:
            return None, True
    return None, False


def _match_vendor(expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual is None
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    return normalize_vendor(expected) == normalize_vendor(actual)


def _match_invoice_number(expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual is None
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    return expected.strip().lower() == actual.strip().lower()


def _match_exact_string(expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual is None
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    return expected == actual


def _match_number(
    expected: Any,
    actual: Any,
    tolerance: float,
) -> tuple[bool, bool]:
    """Compare two optional numbers within tolerance; also flag string-number actuals."""
    if expected is None:
        _, was_str = parse_numeric(actual)
        return actual is None, was_str

    exp_num, _ = parse_numeric(expected)
    act_num, was_str = parse_numeric(actual)
    if exp_num is None or act_num is None:
        return False, was_str
    return abs(exp_num - act_num) <= tolerance, was_str


def _match_description(expected: Any, actual: Any) -> bool:
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    norm_exp = normalize_description(expected)
    norm_act = normalize_description(actual)
    if norm_exp == norm_act:
        return True
    ratio = SequenceMatcher(None, norm_exp, norm_act).ratio()
    return ratio >= DESCRIPTION_SIMILARITY_THRESHOLD


def _evaluate_line_items(
    doc_id: str,
    expected: Any,
    actual: Any,
) -> tuple[bool, LineItemStats, int, list[FieldError]]:
    """Compare line_items in printed order."""
    errors: list[FieldError] = []
    string_failures = 0

    if expected is None:
        if actual is None:
            return True, LineItemStats(), 0, []
        pred_count = len(actual) if isinstance(actual, list) else 1
        stats = LineItemStats(total_rows=pred_count, gt_rows=0, pred_rows=pred_count)
        errors.append(
            FieldError(
                doc_id=doc_id,
                field="line_items",
                expected=None,
                actual=actual,
                note="Non-invoice should have line_items=null",
            )
        )
        return False, stats, 0, errors

    if not isinstance(actual, list):
        gt_count = len(expected)
        stats = LineItemStats(total_rows=gt_count, gt_rows=gt_count, pred_rows=0)
        errors.append(
            FieldError(
                doc_id=doc_id,
                field="line_items",
                expected=f"<{gt_count} items>",
                actual=actual,
                note="Expected a list of line items",
            )
        )
        return False, stats, 0, errors

    gt_count = len(expected)
    pred_count = len(actual)
    # Using max(gt, pred) penalizes both missing and extra rows in row-level accuracy.
    total_rows = max(gt_count, pred_count)
    stats = LineItemStats(total_rows=total_rows, gt_rows=gt_count, pred_rows=pred_count)

    paired = min(gt_count, pred_count)
    for idx in range(paired):
        exp_row = expected[idx]
        act_row = actual[idx]
        if not isinstance(act_row, dict):
            errors.append(
                FieldError(
                    doc_id=doc_id,
                    field=f"line_items[{idx}]",
                    expected=exp_row,
                    actual=act_row,
                    note="Line item is not an object",
                )
            )
            continue

        desc_ok = _match_description(exp_row.get("description"), act_row.get("description"))
        qty_ok, qty_str = _match_number(
            exp_row.get("quantity"), act_row.get("quantity"), QUANTITY_TOLERANCE
        )
        amt_ok, amt_str = _match_number(
            exp_row.get("amount"), act_row.get("amount"), AMOUNT_TOLERANCE
        )
        string_failures += int(qty_str) + int(amt_str)

        if desc_ok:
            stats.desc_correct += 1
        else:
            errors.append(
                FieldError(
                    doc_id=doc_id,
                    field=f"line_items[{idx}].description",
                    expected=exp_row.get("description"),
                    actual=act_row.get("description"),
                )
            )

        if qty_ok:
            stats.qty_correct += 1
        else:
            errors.append(
                FieldError(
                    doc_id=doc_id,
                    field=f"line_items[{idx}].quantity",
                    expected=exp_row.get("quantity"),
                    actual=act_row.get("quantity"),
                )
            )

        if amt_ok:
            stats.amount_correct += 1
        else:
            errors.append(
                FieldError(
                    doc_id=doc_id,
                    field=f"line_items[{idx}].amount",
                    expected=exp_row.get("amount"),
                    actual=act_row.get("amount"),
                )
            )

        if desc_ok and qty_ok and amt_ok:
            stats.correct_rows += 1

    if pred_count < gt_count:
        for idx in range(pred_count, gt_count):
            errors.append(
                FieldError(
                    doc_id=doc_id,
                    field=f"line_items[{idx}]",
                    expected=expected[idx],
                    actual=None,
                    note="Missing line item row",
                )
            )
    elif pred_count > gt_count:
        for idx in range(gt_count, pred_count):
            errors.append(
                FieldError(
                    doc_id=doc_id,
                    field=f"line_items[{idx}]",
                    expected=None,
                    actual=actual[idx],
                    note="Extra line item row",
                )
            )

    doc_exact = (gt_count == pred_count) and (stats.correct_rows == gt_count)
    return doc_exact, stats, string_failures, errors


def evaluate_document(
    doc_id: str,
    expected: dict[str, Any],
    predicted: dict[str, Any],
) -> DocumentEval:
    """Compare one predicted document JSON against its ground-truth JSON."""
    errors: list[FieldError] = []
    field_matches: dict[str, bool] = {}
    string_failures = 0

    needs_review = bool(predicted.get("needs_review", False))
    raw_reasons = predicted.get("review_reasons")
    review_reasons = list(raw_reasons) if isinstance(raw_reasons, list) else []

    # 1. is_invoice (strict boolean match: 1 != True)
    exp_inv = expected.get("is_invoice")
    act_inv = predicted.get("is_invoice")
    inv_ok = isinstance(act_inv, bool) and (act_inv is exp_inv)
    field_matches["is_invoice"] = inv_ok
    if not inv_ok:
        errors.append(FieldError(doc_id, "is_invoice", exp_inv, act_inv))

    # 2. vendor
    vendor_ok = _match_vendor(expected.get("vendor"), predicted.get("vendor"))
    field_matches["vendor"] = vendor_ok
    if not vendor_ok:
        errors.append(FieldError(doc_id, "vendor", expected.get("vendor"), predicted.get("vendor")))

    # 3. invoice_number
    inv_num_ok = _match_invoice_number(
        expected.get("invoice_number"), predicted.get("invoice_number")
    )
    field_matches["invoice_number"] = inv_num_ok
    if not inv_num_ok:
        errors.append(
            FieldError(
                doc_id,
                "invoice_number",
                expected.get("invoice_number"),
                predicted.get("invoice_number"),
            )
        )

    # 4. invoice_date (exact YYYY-MM-DD)
    date_ok = _match_exact_string(expected.get("invoice_date"), predicted.get("invoice_date"))
    field_matches["invoice_date"] = date_ok
    if not date_ok:
        errors.append(
            FieldError(
                doc_id,
                "invoice_date",
                expected.get("invoice_date"),
                predicted.get("invoice_date"),
            )
        )

    # 5. currency (exact uppercase ISO code)
    curr_ok = _match_exact_string(expected.get("currency"), predicted.get("currency"))
    field_matches["currency"] = curr_ok
    if not curr_ok:
        errors.append(
            FieldError(doc_id, "currency", expected.get("currency"), predicted.get("currency"))
        )

    # 6. line_items
    li_exact, li_stats, li_str_fail, li_errors = _evaluate_line_items(
        doc_id, expected.get("line_items"), predicted.get("line_items")
    )
    field_matches["line_items"] = li_exact
    string_failures += li_str_fail
    errors.extend(li_errors)

    # 7. tax_total
    tax_ok, tax_str = _match_number(
        expected.get("tax_total"), predicted.get("tax_total"), AMOUNT_TOLERANCE
    )
    string_failures += int(tax_str)
    field_matches["tax_total"] = tax_ok
    if not tax_ok:
        errors.append(
            FieldError(doc_id, "tax_total", expected.get("tax_total"), predicted.get("tax_total"))
        )

    # 8. total
    total_ok, total_str = _match_number(
        expected.get("total"), predicted.get("total"), AMOUNT_TOLERANCE
    )
    string_failures += int(total_str)
    field_matches["total"] = total_ok
    if not total_ok:
        errors.append(
            FieldError(doc_id, "total", expected.get("total"), predicted.get("total"))
        )

    all_ok = all(field_matches.values())
    return DocumentEval(
        doc_id=doc_id,
        field_matches=field_matches,
        all_fields_correct=all_ok,
        line_item_stats=li_stats,
        needs_review=needs_review,
        review_reasons=review_reasons,
        string_number_failures=string_failures,
        errors=errors,
    )


def evaluate_dataset(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> EvalSummary:
    """Aggregate evaluation metrics across all labelled documents."""
    doc_evals = [evaluate_document(doc_id, gt, pred) for doc_id, gt, pred in pairs]

    all_field_keys = (*SCALAR_FIELDS, "line_items")
    field_correct = {
        key: sum(1 for d in doc_evals if d.field_matches[key]) for key in all_field_keys
    }
    agg_li = LineItemStats(
        total_rows=sum(d.line_item_stats.total_rows for d in doc_evals),
        gt_rows=sum(d.line_item_stats.gt_rows for d in doc_evals),
        pred_rows=sum(d.line_item_stats.pred_rows for d in doc_evals),
        correct_rows=sum(d.line_item_stats.correct_rows for d in doc_evals),
        desc_correct=sum(d.line_item_stats.desc_correct for d in doc_evals),
        qty_correct=sum(d.line_item_stats.qty_correct for d in doc_evals),
        amount_correct=sum(d.line_item_stats.amount_correct for d in doc_evals),
    )

    all_correct_docs = sum(1 for d in doc_evals if d.all_fields_correct)
    needs_review_docs = sum(1 for d in doc_evals if d.needs_review)
    flagged_wrong_docs = sum(
        1 for d in doc_evals if d.needs_review and not d.all_fields_correct
    )
    unflagged_wrong_docs = sum(
        1 for d in doc_evals if (not d.needs_review) and (not d.all_fields_correct)
    )
    flagged_correct_docs = sum(
        1 for d in doc_evals if d.needs_review and d.all_fields_correct
    )
    string_failures = sum(d.string_number_failures for d in doc_evals)

    return EvalSummary(
        total_docs=len(doc_evals),
        field_correct=field_correct,
        all_fields_correct_docs=all_correct_docs,
        line_item_stats=agg_li,
        needs_review_docs=needs_review_docs,
        flagged_wrong_docs=flagged_wrong_docs,
        unflagged_wrong_docs=unflagged_wrong_docs,
        flagged_correct_docs=flagged_correct_docs,
        string_number_failures=string_failures,
        documents=doc_evals,
    )


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{(num / den) * 100:.1f}%"


def format_results_table(summary: EvalSummary) -> str:
    """Format the per-field accuracy table and validation/review-flag stats."""
    n = summary.total_docs
    li = summary.line_item_stats

    rows = [
        ("is_invoice", f"{summary.field_correct['is_invoice']} / {n}", _pct(summary.field_correct["is_invoice"], n)),
        ("vendor", f"{summary.field_correct['vendor']} / {n}", _pct(summary.field_correct["vendor"], n)),
        ("invoice_number", f"{summary.field_correct['invoice_number']} / {n}", _pct(summary.field_correct["invoice_number"], n)),
        ("invoice_date", f"{summary.field_correct['invoice_date']} / {n}", _pct(summary.field_correct["invoice_date"], n)),
        ("currency", f"{summary.field_correct['currency']} / {n}", _pct(summary.field_correct["currency"], n)),
        ("line_items (row-level)", f"{li.correct_rows} / {li.total_rows}", _pct(li.correct_rows, li.total_rows)),
        ("line_items (document exact)", f"{summary.field_correct['line_items']} / {n}", _pct(summary.field_correct["line_items"], n)),
        ("tax_total", f"{summary.field_correct['tax_total']} / {n}", _pct(summary.field_correct["tax_total"], n)),
        ("total", f"{summary.field_correct['total']} / {n}", _pct(summary.field_correct["total"], n)),
        ("All fields correct (document)", f"{summary.all_fields_correct_docs} / {n}", _pct(summary.all_fields_correct_docs, n)),
    ]

    lines = [
        "| Field | Correct / Total | Accuracy |",
        "|---|---|---|",
    ]
    for label, ratio, acc in rows:
        lines.append(f"| {label} | {ratio} | {acc} |")

    lines.extend(
        [
            "",
            "Line-item sub-field accuracy:",
            f"  description : {li.desc_correct} / {li.total_rows} ({_pct(li.desc_correct, li.total_rows)})",
            f"  quantity    : {li.qty_correct} / {li.total_rows} ({_pct(li.qty_correct, li.total_rows)})",
            f"  amount      : {li.amount_correct} / {li.total_rows} ({_pct(li.amount_correct, li.total_rows)})",
            f"  row counts  : predicted={li.pred_rows}, ground_truth={li.gt_rows}",
            "",
            "Validation & review-flag stats:",
            f"  needs_review=true documents : {summary.needs_review_docs} / {n}",
            f"  wrong docs flagged          : {summary.flagged_wrong_docs} / {summary.flagged_wrong_docs + summary.unflagged_wrong_docs}",
            f"  wrong docs unflagged        : {summary.unflagged_wrong_docs}",
            f"  flagged docs actually right : {summary.flagged_correct_docs}",
            f"  string-number schema fails  : {summary.string_number_failures}",
            "",
            "Note: Measured on the 10 labelled documents (1 document = 10 percentage points).",
            "      If any prompt or rule was tuned after seeing these results, record it in REPORT.md's Tuning log.",
        ]
    )
    return "\n".join(lines)


def format_verbose_errors(summary: EvalSummary) -> str:
    """List every wrong field with expected vs actual values."""
    lines: list[str] = []
    for doc in summary.documents:
        if not doc.errors:
            continue
        flag_note = " [needs_review=true]" if doc.needs_review else ""
        lines.append(f"\n{doc.doc_id}{flag_note}:")
        for err in doc.errors:
            suffix = f" ({err.note})" if err.note else ""
            lines.append(
                f"  - {err.field}: expected={err.expected!r} | actual={err.actual!r}{suffix}"
            )
    if not lines:
        return "\nAll fields matched on every evaluated document."
    return "\nDetailed mismatches:" + "\n".join(lines)


def summary_to_dict(
    summary: EvalSummary,
    *,
    source: str,
    timestamp: str,
) -> dict[str, Any]:
    """Serialize evaluation metrics to a machine-readable JSON dict.

    Omits raw confidential document field values so saved summaries stay safe to share.
    """
    n = summary.total_docs
    li = summary.line_item_stats

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den > 0 else None

    return {
        "timestamp": timestamp,
        "source": source,
        "total_docs": n,
        "metrics": {
            "is_invoice": {"correct": summary.field_correct["is_invoice"], "total": n, "accuracy": ratio(summary.field_correct["is_invoice"], n)},
            "vendor": {"correct": summary.field_correct["vendor"], "total": n, "accuracy": ratio(summary.field_correct["vendor"], n)},
            "invoice_number": {"correct": summary.field_correct["invoice_number"], "total": n, "accuracy": ratio(summary.field_correct["invoice_number"], n)},
            "invoice_date": {"correct": summary.field_correct["invoice_date"], "total": n, "accuracy": ratio(summary.field_correct["invoice_date"], n)},
            "currency": {"correct": summary.field_correct["currency"], "total": n, "accuracy": ratio(summary.field_correct["currency"], n)},
            "line_items_row_level": {"correct": li.correct_rows, "total": li.total_rows, "accuracy": ratio(li.correct_rows, li.total_rows)},
            "line_items_document_exact": {"correct": summary.field_correct["line_items"], "total": n, "accuracy": ratio(summary.field_correct["line_items"], n)},
            "tax_total": {"correct": summary.field_correct["tax_total"], "total": n, "accuracy": ratio(summary.field_correct["tax_total"], n)},
            "total": {"correct": summary.field_correct["total"], "total": n, "accuracy": ratio(summary.field_correct["total"], n)},
            "all_fields_correct_document": {"correct": summary.all_fields_correct_docs, "total": n, "accuracy": ratio(summary.all_fields_correct_docs, n)},
        },
        "line_items_subfields": {
            "description": {"correct": li.desc_correct, "total": li.total_rows, "accuracy": ratio(li.desc_correct, li.total_rows)},
            "quantity": {"correct": li.qty_correct, "total": li.total_rows, "accuracy": ratio(li.qty_correct, li.total_rows)},
            "amount": {"correct": li.amount_correct, "total": li.total_rows, "accuracy": ratio(li.amount_correct, li.total_rows)},
            "gt_rows": li.gt_rows,
            "pred_rows": li.pred_rows,
        },
        "validation_stats": {
            "needs_review_docs": summary.needs_review_docs,
            "flagged_wrong_docs": summary.flagged_wrong_docs,
            "unflagged_wrong_docs": summary.unflagged_wrong_docs,
            "flagged_correct_docs": summary.flagged_correct_docs,
            "string_number_failures": summary.string_number_failures,
        },
        "documents": [
            {
                "doc_id": d.doc_id,
                "all_fields_correct": d.all_fields_correct,
                "needs_review": d.needs_review,
                "review_reasons": d.review_reasons,
                "string_number_failures": d.string_number_failures,
                "field_matches": d.field_matches,
                "failed_fields": [e.field for e in d.errors],
                "line_items": {
                    "correct_rows": d.line_item_stats.correct_rows,
                    "total_rows": d.line_item_stats.total_rows,
                    "gt_rows": d.line_item_stats.gt_rows,
                    "pred_rows": d.line_item_stats.pred_rows,
                },
            }
            for d in summary.documents
        ],
    }
