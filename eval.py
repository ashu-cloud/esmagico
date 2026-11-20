"""CLI evaluation harness for the 10 labelled sample documents (FR4, 04_EVAL_RULES.md).

Calls extract_invoice() directly (never through HTTP) when running live, or
evaluates pre-saved prediction JSONs when --predictions-dir is given.

Usage:
    python eval.py                                     # Live run on L01..L10
    python eval.py --predictions-dir baselines/phase2_raw -v
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invoice_extractor.config import (
    LOGS_DIR,
    REPO_ROOT,
    get_gemini_model,
    get_labelled_dir,
)
from invoice_extractor.evaluator import (
    evaluate_dataset,
    format_results_table,
    format_verbose_errors,
    summary_to_dict,
)
from invoice_extractor.extractor import extract_invoice
from invoice_extractor.loader import SUPPORTED_EXTENSIONS

EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"


def _load_labelled_ground_truth(labelled_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    """Return sorted (doc_id, doc_file_path, gt_json_dict) tuples from labelled/."""
    if not labelled_dir.exists():
        raise FileNotFoundError(f"Labelled directory not found: {labelled_dir}")

    doc_files = sorted(
        p
        for p in labelled_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not doc_files:
        raise FileNotFoundError(f"No supported documents found in {labelled_dir}")

    entries: list[tuple[str, Path, dict[str, Any]]] = []
    for doc_path in doc_files:
        gt_path = labelled_dir / f"{doc_path.stem}.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground-truth JSON for {doc_path.name}: {gt_path}")
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        entries.append((doc_path.stem, doc_path, gt_data))
    return entries


def _load_saved_predictions(
    entries: list[tuple[str, Path, dict[str, Any]]],
    predictions_dir: Path,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Load <DOC_ID>.json predictions from a directory for offline evaluation."""
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for doc_id, _, gt_data in entries:
        pred_path = predictions_dir / f"{doc_id}.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file for {doc_id}: {pred_path}")
        pred_data = json.loads(pred_path.read_text(encoding="utf-8"))
        pairs.append((doc_id, gt_data, pred_data))
    return pairs


def _run_live_extraction(
    entries: list[tuple[str, Path, dict[str, Any]]],
    *,
    model: str,
    log_file: Path,
    save_predictions_dir: Path | None,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Run extract_invoice() directly on each labelled document (no HTTP)."""
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)

    for idx, (doc_id, doc_path, gt_data) in enumerate(entries, start=1):
        print(f"[{idx:02d}/{len(entries):02d}] Extracting {doc_path.name}...", flush=True)
        try:
            result = extract_invoice(doc_path, model=model, log_file=log_file)
            pred_data = result.to_dict()
        except Exception as exc:
            pred_data = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "needs_review": True,
                "review_reasons": [f"Extraction error: {type(exc).__name__}"],
            }
            print(
                f"  -> FAILED ({type(exc).__name__}: {exc})",
                file=sys.stderr,
                flush=True,
            )

        if save_predictions_dir is not None:
            out_path = save_predictions_dir / f"{doc_id}.json"
            out_path.write_text(
                json.dumps(pred_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        pairs.append((doc_id, gt_data, pred_data))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate invoice extraction accuracy against the 10 labelled documents."
    )
    parser.add_argument(
        "--sample-pack",
        default=None,
        help="Path to the sample pack folder (defaults to SAMPLE_PACK_DIR / 'Sample Pack').",
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Evaluate pre-saved <DOC_ID>.json files in this directory instead of calling the LLM.",
    )
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="When running live, also save the 10 predicted JSONs to this directory.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="When running live, override GEMINI_MODEL from .env.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="When running live, token usage JSONL log path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the machine-readable eval results JSON (default: timestamped file in eval_results/).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print every mismatched field with expected vs actual values.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    try:
        labelled_dir = get_labelled_dir(args.sample_pack)
        entries = _load_labelled_ground_truth(labelled_dir)

        if args.predictions_dir:
            pred_dir = Path(args.predictions_dir)
            pairs = _load_saved_predictions(entries, pred_dir)
            source = f"predictions:{pred_dir}"
        else:
            model = get_gemini_model(args.model)
            log_file = (
                Path(args.log_file)
                if args.log_file
                else LOGS_DIR / f"eval_{stamp}.jsonl"
            )
            save_dir = Path(args.save_predictions) if args.save_predictions else None
            print(f"Running live evaluation on {len(entries)} labelled docs with model={model}")
            print(f"Token log: {log_file}")
            pairs = _run_live_extraction(
                entries,
                model=model,
                log_file=log_file,
                save_predictions_dir=save_dir,
            )
            source = f"live:{model}"
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = evaluate_dataset(pairs)
    print()
    print(format_results_table(summary))
    if args.verbose:
        print(format_verbose_errors(summary))

    out_path = (
        Path(args.output)
        if args.output
        else EVAL_RESULTS_DIR / f"eval_{stamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary_to_dict(summary, source=source, timestamp=now.isoformat())
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved evaluation results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
