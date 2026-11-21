"""Robustness and multi-run stability evaluation (separate from main eval.py).

Reuses invoice_extractor.extractor.extract_invoice() and
invoice_extractor.evaluator.evaluate_dataset() without modifying the main eval,
prompt, or validation logic.

Usage:
    python robustness_eval.py [--seed 42] [--stability-runs 3] [--delay 0.5]
    python robustness_eval.py --degradations jpeg_compression partial_crop --stability-runs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_degraded import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEGRADATION_TYPES,
    generate_degraded_dataset,
)
from invoice_extractor.config import (
    LOGS_DIR,
    REPO_ROOT,
    get_api_backoff_base_seconds,
    get_api_max_retries,
    get_api_timeout_seconds,
    get_gemini_model,
    get_labelled_dir,
)
from invoice_extractor.evaluator import (
    EvalSummary,
    evaluate_dataset,
    format_results_table,
    format_verbose_errors,
    summary_to_dict,
)
from invoice_extractor.extractor import extract_invoice
from invoice_extractor.loader import SUPPORTED_EXTENSIONS

EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"


def _load_ground_truth(labelled_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    doc_files = sorted(
        p
        for p in labelled_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    entries: list[tuple[str, Path, dict[str, Any]]] = []
    for doc_path in doc_files:
        gt_path = labelled_dir / f"{doc_path.stem}.json"
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        entries.append((doc_path.stem, doc_path, gt_data))
    return entries


def _is_outage_failure(pred: dict[str, Any]) -> bool:
    """Return True if a prediction record represents an upstream provider outage/timeout."""
    reasons = pred.get("review_reasons") or []
    return any(
        isinstance(r, str) and r.startswith("Extraction error: Provider")
        for r in reasons
    )


def _run_extractor_safe(
    doc_path: Path,
    *,
    document_name: str,
    model: str,
    log_file: Path,
) -> dict[str, Any]:
    """Call extract_invoice() directly and convert any exception to a flagged record."""
    try:
        res = extract_invoice(
            doc_path,
            model=model,
            log_file=log_file,
            document_name=document_name,
        )
        return res.to_dict()
    except Exception as exc:
        return {
            "is_invoice": None,
            "vendor": None,
            "invoice_number": None,
            "invoice_date": None,
            "currency": None,
            "line_items": None,
            "tax_total": None,
            "total": None,
            "needs_review": True,
            "review_reasons": [f"Extraction error: {type(exc).__name__}: {exc}"],
        }


def _review_flag_metrics(summary: EvalSummary) -> dict[str, Any]:
    """Compute review-flag breakdown (all docs)."""
    total_wrong = summary.flagged_wrong_docs + summary.unflagged_wrong_docs
    total_flagged = summary.needs_review_docs
    return {
        "needs_review_docs": total_flagged,
        "total_wrong_docs": total_wrong,
        "wrong_docs_flagged": summary.flagged_wrong_docs,
        "wrong_docs_unflagged": summary.unflagged_wrong_docs,
        "flagged_docs_actually_wrong": summary.flagged_wrong_docs,
        "flagged_docs_actually_correct": summary.flagged_correct_docs,
        "recall_of_flag_on_wrong": (
            round(summary.flagged_wrong_docs / total_wrong, 4) if total_wrong > 0 else None
        ),
        "precision_of_flag_on_wrong": (
            round(summary.flagged_wrong_docs / total_flagged, 4) if total_flagged > 0 else None
        ),
    }


def _evaluate_with_outage_split(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    source: str,
    timestamp: str,
) -> tuple[EvalSummary, EvalSummary | None, dict[str, Any]]:
    """Evaluate both end-to-end and conditional on a successful API response."""
    e2e_summary = evaluate_dataset(pairs)
    outage_ids = [doc_id for doc_id, _, pred in pairs if _is_outage_failure(pred)]
    ok_pairs = [(doc_id, gt, pred) for doc_id, gt, pred in pairs if not _is_outage_failure(pred)]

    cond_summary = evaluate_dataset(ok_pairs) if ok_pairs else None

    payload = summary_to_dict(e2e_summary, source=source, timestamp=timestamp)
    payload["review_flag_summary"] = _review_flag_metrics(e2e_summary)
    payload["outage_exhausted_count"] = len(outage_ids)
    payload["outage_exhausted_docs"] = outage_ids
    if cond_summary is not None:
        cond_dict = summary_to_dict(
            cond_summary, source=f"{source}:conditional_on_api_success", timestamp=timestamp
        )
        cond_dict["review_flag_summary_excluding_outages"] = _review_flag_metrics(cond_summary)
        payload["conditional_on_api_success"] = cond_dict
    else:
        payload["conditional_on_api_success"] = None

    return e2e_summary, cond_summary, payload


def _diff_runs(
    doc_id: str,
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare all fields of `doc_id` across repeated clean runs."""
    diffs: list[dict[str, Any]] = []
    if len(runs) < 2:
        return diffs

    base = runs[0]
    all_keys = sorted(set().union(*(r.keys() for r in runs)))
    for key in all_keys:
        values = [r.get(key) for r in runs]
        if any(v != base.get(key) for v in values[1:]):
            diffs.append(
                {
                    "doc_id": doc_id,
                    "field": key,
                    "values_by_run": {f"run_{i + 1}": v for i, v in enumerate(values)},
                }
            )
    return diffs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run degraded-image robustness eval and multi-run clean stability eval."
    )
    parser.add_argument(
        "--sample-pack",
        default=None,
        help="Path to the sample pack folder (defaults to SAMPLE_PACK_DIR / 'Sample Pack').",
    )
    parser.add_argument(
        "--degraded-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Git-ignored folder holding degraded copies and local detailed logs.",
    )
    parser.add_argument(
        "--degradations",
        nargs="*",
        choices=list(DEGRADATION_TYPES),
        default=list(DEGRADATION_TYPES),
        help="Subset of degradations to run (default: all 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Fixed RNG seed for reproducible degradations (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=3,
        help="Number of repeated runs on the clean labelled set (default: 3).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Seconds to pause between API calls.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override GEMINI_MODEL from .env.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    model = get_gemini_model(args.model)
    retry_cfg = {
        "api_timeout_seconds": get_api_timeout_seconds(),
        "api_max_retries": get_api_max_retries(),
        "api_backoff_base_seconds": get_api_backoff_base_seconds(),
        "inter_call_delay_seconds": args.delay,
    }
    degraded_root = Path(args.degraded_dir)
    labelled_dir = get_labelled_dir(args.sample_pack)

    # Generate reproducible degraded dataset (or refresh deterministically from the fixed seed).
    manifest_path = generate_degraded_dataset(
        sample_pack=args.sample_pack,
        output_dir=degraded_root,
        seed=args.seed,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gt_entries = _load_ground_truth(labelled_dir)

    rob_log_file = LOGS_DIR / f"robustness_{stamp}.jsonl"
    stab_log_file = LOGS_DIR / f"stability_{stamp}.jsonl"

    print(
        f"=== Robustness Evaluation (model={model}, seed={args.seed}, "
        f"degradations={args.degradations}) ===",
        flush=True,
    )
    print(f"Retry config         : {retry_cfg}", flush=True)
    print(f"Degraded dataset dir : {degraded_root}", flush=True)
    print(f"Robustness token log : {rob_log_file}", flush=True)

    degradation_summaries: dict[str, Any] = {}
    degradation_verbose: dict[str, Any] = {}
    predictions_root = degraded_root / "predictions" / stamp

    # 1. Run extractor on selected degraded sets.
    for deg in args.degradations:
        print(f"\n--- Degradation: {deg} ---", flush=True)
        deg_preds_dir = predictions_root / deg
        deg_preds_dir.mkdir(parents=True, exist_ok=True)
        pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        for idx, (doc_id, _, gt_data) in enumerate(gt_entries, start=1):
            rel_out = manifest["degradations"][deg][doc_id]["output"]
            deg_path = degraded_root / rel_out
            pred = _run_extractor_safe(
                deg_path,
                document_name=f"{deg}/{deg_path.name}",
                model=model,
                log_file=rob_log_file,
            )
            (deg_preds_dir / f"{doc_id}.json").write_text(
                json.dumps(pred, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            flag = " [NEEDS_REVIEW]" if pred.get("needs_review") else ""
            outage = " [OUTAGE]" if _is_outage_failure(pred) else ""
            print(
                f"  [{idx:02d}/10] {deg}/{deg_path.name} -> is_invoice={pred.get('is_invoice')}{flag}{outage}",
                flush=True,
            )
            pairs.append((doc_id, gt_data, pred))
            if args.delay > 0:
                time.sleep(args.delay)

        e2e_summary, cond_summary, entry_payload = _evaluate_with_outage_split(
            pairs,
            source=f"degraded:{deg}:seed={args.seed}",
            timestamp=now.isoformat(),
        )
        print(f"\nEnd-to-end results for [{deg}]:", flush=True)
        print(format_results_table(e2e_summary), flush=True)
        print(format_verbose_errors(e2e_summary), flush=True)
        print(
            f"\nOutage-exhausted docs [{deg}]: {entry_payload['outage_exhausted_count']} "
            f"({entry_payload['outage_exhausted_docs']})",
            flush=True,
        )
        if cond_summary is not None:
            print(
                f"Conditional on API success [{deg}] ({cond_summary.total_docs}/10 docs): "
                f"doc_exact={cond_summary.all_fields_correct_docs}/{cond_summary.total_docs}, "
                f"rows={cond_summary.line_item_stats.correct_rows}/{cond_summary.line_item_stats.total_rows}",
                flush=True,
            )

        degradation_summaries[deg] = entry_payload
        degradation_verbose[deg] = {
            "outage_exhausted_docs": entry_payload["outage_exhausted_docs"],
            "review_flag_summary": entry_payload["review_flag_summary"],
            "review_flag_summary_excluding_outages": (
                entry_payload["conditional_on_api_success"]["review_flag_summary_excluding_outages"]
                if entry_payload["conditional_on_api_success"]
                else None
            ),
            "errors": [
                {
                    "doc_id": e.doc_id,
                    "field": e.field,
                    "expected": e.expected,
                    "actual": e.actual,
                    "note": e.note,
                    "needs_review": doc_ev.needs_review,
                    "review_reasons": doc_ev.review_reasons,
                    "is_outage": doc_ev.doc_id in entry_payload["outage_exhausted_docs"],
                }
                for doc_ev in e2e_summary.documents
                for e in doc_ev.errors
            ],
        }

    # 2. Clean labelled set repeated runs (stability check).
    print(f"\n=== Clean Stability Evaluation ({args.stability_runs} runs on L01..L10) ===", flush=True)
    print(f"Stability token log  : {stab_log_file}", flush=True)
    stability_preds_root = degraded_root / "stability_runs" / stamp
    runs_by_doc: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id, _, _ in gt_entries}
    stability_run_summaries: list[dict[str, Any]] = []

    for run_idx in range(1, args.stability_runs + 1):
        print(f"\n--- Clean Run {run_idx}/{args.stability_runs} ---", flush=True)
        run_dir = stability_preds_root / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pairs = []

        for doc_idx, (doc_id, doc_path, gt_data) in enumerate(gt_entries, start=1):
            pred = _run_extractor_safe(
                doc_path,
                document_name=f"clean_run{run_idx}/{doc_path.name}",
                model=model,
                log_file=stab_log_file,
            )
            (run_dir / f"{doc_id}.json").write_text(
                json.dumps(pred, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            runs_by_doc[doc_id].append(pred)
            pairs.append((doc_id, gt_data, pred))
            outage = " [OUTAGE]" if _is_outage_failure(pred) else ""
            print(
                f"  [run {run_idx} | {doc_idx:02d}/10] {doc_path.name} -> "
                f"is_invoice={pred.get('is_invoice')} needs_review={pred.get('needs_review')}{outage}",
                flush=True,
            )
            if args.delay > 0:
                time.sleep(args.delay)

        e2e_run, cond_run, run_payload = _evaluate_with_outage_split(
            pairs,
            source=f"clean_stability_run_{run_idx}:{model}",
            timestamp=now.isoformat(),
        )
        print(format_results_table(e2e_run), flush=True)
        print(
            f"Run {run_idx} outage-exhausted: {run_payload['outage_exhausted_count']} "
            f"({run_payload['outage_exhausted_docs']})",
            flush=True,
        )
        stability_run_summaries.append(run_payload)

    # Compare every field across repeated clean runs (all runs AND non-outage runs only).
    all_run_diffs: list[dict[str, Any]] = []
    non_outage_run_diffs: list[dict[str, Any]] = []
    for doc_id, _, _ in gt_entries:
        all_run_diffs.extend(_diff_runs(doc_id, runs_by_doc[doc_id]))
        ok_runs = [r for r in runs_by_doc[doc_id] if not _is_outage_failure(r)]
        non_outage_run_diffs.extend(_diff_runs(doc_id, ok_runs))

    print("\n=== Stability Diff Across Clean Runs ===", flush=True)
    print(f"Total diffs (all runs)             : {len(all_run_diffs)}", flush=True)
    print(f"Total diffs (successful API runs)  : {len(non_outage_run_diffs)}", flush=True)
    for d in all_run_diffs:
        print(f"  - {d['doc_id']}.{d['field']}: {d['values_by_run']}", flush=True)

    # 3. Save timestamped results files without overwriting any earlier runs.
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rob_out = EVAL_RESULTS_DIR / f"robustness_{stamp}.json"
    rob_payload = {
        "timestamp": now.isoformat(),
        "model": model,
        "seed": args.seed,
        "retry_config": retry_cfg,
        "token_log": str(rob_log_file.relative_to(REPO_ROOT)),
        "degradations": degradation_summaries,
    }
    rob_out.write_text(
        json.dumps(rob_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stab_out = EVAL_RESULTS_DIR / f"stability_{stamp}.json"
    stab_payload = {
        "timestamp": now.isoformat(),
        "model": model,
        "num_runs": args.stability_runs,
        "retry_config": retry_cfg,
        "token_log": str(stab_log_file.relative_to(REPO_ROOT)),
        "runs": stability_run_summaries,
        "total_field_diffs_across_runs": len(all_run_diffs),
        "total_field_diffs_among_successful_runs": len(non_outage_run_diffs),
        "differing_fields": [
            {"doc_id": d["doc_id"], "field": d["field"]} for d in all_run_diffs
        ],
        "differing_fields_among_successful_runs": [
            {"doc_id": d["doc_id"], "field": d["field"]} for d in non_outage_run_diffs
        ],
    }
    stab_out.write_text(
        json.dumps(stab_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    local_reports_dir = degraded_root / "reports"
    local_reports_dir.mkdir(parents=True, exist_ok=True)
    verbose_out = local_reports_dir / f"robustness_and_stability_verbose_{stamp}.json"
    verbose_out.write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(),
                "model": model,
                "seed": args.seed,
                "retry_config": retry_cfg,
                "degradations_verbose": degradation_verbose,
                "stability_diffs": all_run_diffs,
                "stability_diffs_among_successful_runs": non_outage_run_diffs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved robustness summary : {rob_out}", flush=True)
    print(f"Saved stability summary  : {stab_out}", flush=True)
    print(f"Saved local verbose log  : {verbose_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
