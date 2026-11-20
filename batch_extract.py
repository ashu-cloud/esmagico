"""Batch CLI to extract structured JSON for all documents in the sample pack.

Usage:
    python batch_extract.py [--sample-pack PATH] [--output-dir DIR]
                            [--snapshot-dir DIR] [--model MODEL]
                            [--subset all|labelled|unlabelled] [--log-file PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from invoice_extractor.config import (
    LOGS_DIR,
    OUTPUTS_DIR,
    get_gemini_model,
    get_labelled_dir,
    get_unlabelled_dir,
)
from invoice_extractor.extractor import extract_invoice
from invoice_extractor.loader import SUPPORTED_EXTENSIONS
from invoice_extractor.token_logger import log_token_usage


def discover_documents(
    sample_pack: str | Path | None,
    subset: str = "all",
) -> list[Path]:
    """Find all supported document files in labelled/ and/or unlabelled/."""
    folders: list[Path] = []
    if subset in {"all", "labelled"}:
        folders.append(get_labelled_dir(sample_pack))
    if subset in {"all", "unlabelled"}:
        folders.append(get_unlabelled_dir(sample_pack))

    docs: list[Path] = []
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        folder_docs = sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        docs.extend(folder_docs)
    return docs


def _default_log_file() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return LOGS_DIR / f"batch_{stamp}.jsonl"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run invoice extraction across all sample pack documents."
    )
    parser.add_argument(
        "--sample-pack",
        default=None,
        help="Path to the sample pack folder (defaults to SAMPLE_PACK_DIR / 'Sample Pack').",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR),
        help="Directory to write <DOC_ID>.json files (default: outputs/).",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Optional separate folder to also save an immutable copy of this run.",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "labelled", "unlabelled"],
        default="all",
        help="Which subset of the sample pack to process (default: all).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override GEMINI_MODEL from .env.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to the JSONL token usage log file (default: timestamped file in logs/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chosen_model = get_gemini_model(args.model)
    output_dir = Path(args.output_dir)
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    log_file = Path(args.log_file) if args.log_file else _default_log_file()

    try:
        documents = discover_documents(args.sample_pack, subset=args.subset)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not documents:
        print("Error: no supported documents found.", file=sys.stderr)
        return 1

    print(f"Running {len(documents)} document(s) with model={chosen_model}")
    print(f"Output dir: {output_dir}")
    if snapshot_dir:
        print(f"Snapshot dir: {snapshot_dir}")
    print(f"Token log: {log_file}")

    succeeded = 0
    failed = 0

    for idx, doc_path in enumerate(documents, start=1):
        out_name = f"{doc_path.stem}.json"
        try:
            result = extract_invoice(
                doc_path,
                model=chosen_model,
                log_file=log_file,
            )
            payload = result.to_dict()
            succeeded += 1
            rows = len(payload["line_items"]) if payload["line_items"] is not None else None
            flag = " [NEEDS_REVIEW]" if payload.get("needs_review") else ""
            print(
                f"[{idx:02d}/{len(documents):02d}] {doc_path.name} -> {out_name} "
                f"(OK, is_invoice={payload['is_invoice']}, rows={rows}{flag})",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            # Record failures explicitly in output JSON and token log instead of hiding them.
            payload = {
                "document": doc_path.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            log_token_usage(
                document=doc_path.name,
                model=chosen_model,
                input_tokens=0,
                output_tokens=0,
                attempt=1,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                log_file=log_file,
            )
            print(
                f"[{idx:02d}/{len(documents):02d}] {doc_path.name} -> {out_name} "
                f"(FAILED: {type(exc).__name__}: {exc})",
                file=sys.stderr,
                flush=True,
            )

        _write_json(output_dir / out_name, payload)
        if snapshot_dir is not None:
            _write_json(snapshot_dir / out_name, payload)

    print(f"Done: {succeeded} succeeded, {failed} failed (total {len(documents)}).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
