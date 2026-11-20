"""Token usage and USD cost analysis from JSONL token logs (FR6, Phase 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PRICING_DATE_CHECKED = "2026-11-20"
PRICING_SOURCE = "Google Gemini API Pricing (ai.google.dev/gemini-api/docs/pricing)"

# Paid-tier list prices in USD per 1,000,000 tokens (output includes thinking tokens).
MODEL_PRICES_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 7.50),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.8-flash": (0.75, 3.75),  # Introductory rate through 2026-12-31
}


@dataclass(frozen=True)
class CostReport:
    """Aggregated token and USD cost metrics for one log file."""

    log_file: str
    model: str
    input_price_per_1m: float
    output_price_per_1m: float
    num_documents: int
    total_calls: int
    first_attempt_calls: int
    retry_calls: int
    error_calls: int
    first_input_tokens: int
    first_output_tokens: int
    retry_input_tokens: int
    retry_output_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    first_attempt_cost_usd: float
    retry_cost_usd: float
    total_cost_usd: float
    avg_input_tokens_per_doc: float
    avg_output_tokens_per_doc: float
    avg_cost_per_doc_usd: float
    retry_cost_share_pct: float
    cost_per_1000_docs_usd: float


def _tokens_to_usd(
    input_tokens: int,
    output_tokens: int,
    in_price: float,
    out_price: float,
) -> float:
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


def compute_cost_report(
    log_file: str | Path,
    *,
    model_filter: str | None = None,
    input_price_per_1m: float | None = None,
    output_price_per_1m: float | None = None,
) -> CostReport:
    """Parse a JSONL token log and compute average tokens and USD cost per document."""
    path = Path(log_file)
    if not path.exists():
        raise FileNotFoundError(f"Token log file not found: {path}")

    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if model_filter and rec.get("model") != model_filter:
            continue
        records.append(rec)

    if not records:
        raise ValueError(f"No token log records found in {path}")

    models_seen = sorted({str(r.get("model", "")) for r in records if r.get("model")})
    if len(models_seen) > 1 and model_filter is None:
        raise ValueError(
            f"Log file contains multiple models ({', '.join(models_seen)}). "
            "Pass --model to compute cost for one model."
        )
    model = models_seen[0]

    if input_price_per_1m is None or output_price_per_1m is None:
        if model not in MODEL_PRICES_USD_PER_1M:
            raise ValueError(
                f"No built-in price table entry for '{model}'. "
                "Provide --input-price and --output-price."
            )
        default_in, default_out = MODEL_PRICES_USD_PER_1M[model]
        in_price = input_price_per_1m if input_price_per_1m is not None else default_in
        out_price = output_price_per_1m if output_price_per_1m is not None else default_out
    else:
        in_price = input_price_per_1m
        out_price = output_price_per_1m

    unique_docs = {str(r.get("document", f"doc_{i}")) for i, r in enumerate(records)}
    num_docs = len(unique_docs)

    first_records = [r for r in records if int(r.get("attempt", 1)) == 1]
    retry_records = [r for r in records if int(r.get("attempt", 1)) > 1]
    error_calls = sum(1 for r in records if r.get("status") != "ok")

    first_in = sum(int(r.get("input_tokens", 0)) for r in first_records)
    first_out = sum(int(r.get("output_tokens", 0)) for r in first_records)
    retry_in = sum(int(r.get("input_tokens", 0)) for r in retry_records)
    retry_out = sum(int(r.get("output_tokens", 0)) for r in retry_records)

    total_in = first_in + retry_in
    total_out = first_out + retry_out

    first_cost = _tokens_to_usd(first_in, first_out, in_price, out_price)
    retry_cost = _tokens_to_usd(retry_in, retry_out, in_price, out_price)
    total_cost = first_cost + retry_cost

    avg_in = total_in / num_docs
    avg_out = total_out / num_docs
    avg_cost = total_cost / num_docs
    retry_share = (retry_cost / total_cost * 100.0) if total_cost > 0 else 0.0

    return CostReport(
        log_file=str(path),
        model=model,
        input_price_per_1m=in_price,
        output_price_per_1m=out_price,
        num_documents=num_docs,
        total_calls=len(records),
        first_attempt_calls=len(first_records),
        retry_calls=len(retry_records),
        error_calls=error_calls,
        first_input_tokens=first_in,
        first_output_tokens=first_out,
        retry_input_tokens=retry_in,
        retry_output_tokens=retry_out,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        first_attempt_cost_usd=first_cost,
        retry_cost_usd=retry_cost,
        total_cost_usd=total_cost,
        avg_input_tokens_per_doc=avg_in,
        avg_output_tokens_per_doc=avg_out,
        avg_cost_per_doc_usd=avg_cost,
        retry_cost_share_pct=retry_share,
        cost_per_1000_docs_usd=avg_cost * 1000.0,
    )


def format_cost_report(report: CostReport) -> str:
    """Format a human-readable cost summary table."""
    lines = [
        f"Cost Summary ({report.log_file})",
        "-" * 60,
        f"Model                    : {report.model}",
        f"List price (per 1M tok)  : ${report.input_price_per_1m:.2f} input / ${report.output_price_per_1m:.2f} output",
        f"Pricing checked          : {PRICING_DATE_CHECKED} ({PRICING_SOURCE})",
        f"Documents processed      : {report.num_documents}",
        f"API calls (1st / retry)  : {report.total_calls} ({report.first_attempt_calls} first-attempt, {report.retry_calls} validation retry, {report.error_calls} errors)",
        "",
        "Token usage (totals):",
        f"  First-attempt tokens   : {report.first_input_tokens:,} in / {report.first_output_tokens:,} out",
        f"  Retry tokens           : {report.retry_input_tokens:,} in / {report.retry_output_tokens:,} out",
        f"  Total tokens           : {report.total_input_tokens:,} in / {report.total_output_tokens:,} out",
        "",
        "Per-document averages:",
        f"  Avg input tokens / doc : {report.avg_input_tokens_per_doc:,.1f}",
        f"  Avg output tokens / doc: {report.avg_output_tokens_per_doc:,.1f}",
        f"  Avg cost / doc (USD)   : ${report.avg_cost_per_doc_usd:.6f}",
        "",
        "Cost breakdown (USD, paid-tier list price):",
        f"  First-attempt cost     : ${report.first_attempt_cost_usd:.6f}",
        f"  Retry cost             : ${report.retry_cost_usd:.6f} ({report.retry_cost_share_pct:.1f}% of total)",
        f"  Total run cost         : ${report.total_cost_usd:.6f}",
        f"  Est. cost / 1,000 docs : ${report.cost_per_1000_docs_usd:.2f}",
        "",
        "Method: Token counts read directly from Gemini API usage_metadata across all",
        "        calls (including retries), multiplied by paid-tier list price per token,",
        "        and averaged over unique documents processed.",
    ]
    return "\n".join(lines)
