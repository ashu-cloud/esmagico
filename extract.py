"""Thin CLI wrapper for single-document invoice extraction (secondary interface).

Usage:
    python extract.py <file> [--model MODEL] [--log-file PATH]

Exit codes:
    0  Extraction succeeded (including is_invoice=false and needs_review=true)
    1  Configuration or unexpected error
    2  Bad input file (missing, unsupported type, empty, or corrupt)
    3  Upstream provider error or timeout
"""

from __future__ import annotations

import argparse
import json
import sys

from invoice_extractor.extractor import (
    ProviderError,
    ProviderTimeoutError,
    extract_invoice,
)
from invoice_extractor.loader import InvalidDocumentError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured invoice JSON from a PDF or image file."
    )
    parser.add_argument(
        "file",
        help="Path to a PDF, JPG, or PNG document.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override GEMINI_MODEL from .env (default: gemini-3.5-flash-lite).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to the JSONL token usage log file (default: logs/token_usage.jsonl).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = extract_invoice(
            args.file,
            model=args.model,
            log_file=args.log_file,
        )
    except (FileNotFoundError, InvalidDocumentError) as exc:
        print(f"Bad input error: {exc}", file=sys.stderr)
        return 2
    except ProviderTimeoutError as exc:
        print(f"Provider timeout error: {exc}", file=sys.stderr)
        return 3
    except ProviderError as exc:
        print(f"Provider error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
