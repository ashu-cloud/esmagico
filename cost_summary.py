"""CLI to summarize token usage and USD cost per document from a JSONL log (Phase 6).

Usage:
    python cost_summary.py logs/final_run_tokens.jsonl
"""

from __future__ import annotations

import argparse
import sys

from invoice_extractor.cost import compute_cost_report, format_cost_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute average tokens and USD cost per document from a token log."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="logs/final_run_tokens.jsonl",
        help="Path to a JSONL token log file (default: logs/final_run_tokens.jsonl).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to a single model if the log contains multiple models.",
    )
    parser.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="Override input price per 1M tokens in USD.",
    )
    parser.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="Override output price per 1M tokens in USD.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compute_cost_report(
            args.log_file,
            model_filter=args.model,
            input_price_per_1m=args.input_price,
            output_price_per_1m=args.output_price,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_cost_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
