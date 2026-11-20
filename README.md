# Invoice Extraction Pipeline

An LLM-based pipeline that reads invoice documents (digital PDFs, scanned PDFs, and JPG/PNG images) as page images and outputs validated, structured JSON using the Google Gemini API.

FastAPI (`api.py`) is the primary interface; a thin CLI (`extract.py`), batch runner (`batch_extract.py`), and evaluation harness (`eval.py`) all call the same framework-independent `invoice_extractor.extractor.extract_invoice` function directly without going through HTTP.

## Requirements

- Python 3.11+
- A Google Gemini API key (from Google AI Studio)
- Local copy of the sample pack folder (kept outside git)

## Setup (from a fresh clone)

1. Create and activate a virtual environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```powershell
   Copy-Item .env.example .env
   ```
   *(On macOS / Linux: `cp .env.example .env`)*

   Edit `.env` and set:
   - `GEMINI_API_KEY` — your Gemini API key (read on the server only; never logged or returned)
   - `GEMINI_MODEL` — Gemini model ID (default: `gemini-3.5-flash-lite`)
   - `SAMPLE_PACK_DIR` — path to your local sample pack directory (defaults to `Sample Pack`)
   - `MAX_UPLOAD_SIZE_MB` — maximum accepted upload size for `POST /extract` in MB (default: `20`)
   - `API_TIMEOUT_SECONDS` — request timeout for upstream Gemini calls in seconds (default: `45`)
   - `API_MAX_RETRIES` — max transient retries with exponential backoff on API errors/timeouts (default: `2`)

## Repository Layout

```text
.
├── .env.example              # Environment variable template
├── .gitignore                # Excludes .env, Sample Pack/, caches, virtualenvs
├── requirements.txt          # Python dependencies
├── api.py                    # Primary FastAPI service (/, POST /extract, GET /health, /docs)
├── static/
│   └── index.html            # Minimal side-by-side upload & review frontend (Bonus)
├── extract.py                # Thin CLI wrapper for single-document extraction
├── batch_extract.py          # Batch runner calling extract_invoice() directly
├── eval.py                   # Evaluation harness over L01..L10 (04_EVAL_RULES.md)
├── cost_summary.py           # Token and USD cost summary CLI
├── REPORT.md                 # Completed evaluation, failure analysis, and cost report
├── invoice_extractor/
│   ├── __init__.py
│   ├── config.py             # Configurable paths, upload limit, timeout, and env loading
│   ├── models.py             # Pydantic models mirroring schema.json
│   ├── loader.py             # PDF page rendering (200 DPI) + EXIF-aware JPG/PNG loader
│   ├── prompt.py             # System prompt (D8) and validation-retry feedback prompt
│   ├── validator.py          # Strict schema + arithmetic validation (0.05 tolerance)
│   ├── token_logger.py       # Append-only JSONL token logger (tracks retries)
│   ├── extractor.py          # Reusable Gemini extractor (backoff + validate + retry/flag)
│   ├── evaluator.py          # Per-field & line-item matching rules and reporting
│   └── cost.py               # Paid list-price cost calculation & retry split
├── outputs/                  # Final extracted JSON for all 20 sample documents
├── baselines/                # Saved raw Phase 2 baseline for before/after comparison
├── eval_results/             # Timestamped machine-readable eval results (*.json)
└── logs/                     # Per-run JSONL token usage logs (*.jsonl)
```

## Usage

### 1. Start the FastAPI server (primary interface)

```bash
uvicorn api:app --reload
```

- Web UI (upload a document and inspect extracted fields side-by-side): `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Interactive Swagger docs (try API requests in the browser): `http://127.0.0.1:8000/docs`

Example request with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "file=@Sample Pack/labelled/L01.pdf"
```

HTTP status codes:
- `200` — Extraction JSON returned (including `is_invoice: false` and `needs_review: true` results).
- `400` — Bad input (unsupported file type, empty file, file over `MAX_UPLOAD_SIZE_MB`, or corrupt PDF/image).
- `502` / `504` — Upstream Gemini API error (`502`) or timeout (`504`).

### 2. Single-document CLI (secondary wrapper)

```bash
python extract.py "Sample Pack/labelled/L01.pdf"
```

Optional flags:
- `--model <model_id>` — override `GEMINI_MODEL` from `.env`
- `--log-file <path>` — write token usage to a specific JSONL file (defaults to `logs/token_usage.jsonl`)

CLI exit codes: `0` success, `1` config/unexpected error, `2` bad input file, `3` upstream provider error/timeout.

### 3. Batch extraction (all 20 sample documents)

```bash
python batch_extract.py --output-dir outputs --log-file logs/final_run_tokens.jsonl
```

Optional flags:
- `--sample-pack <dir>` — override `SAMPLE_PACK_DIR`
- `--output-dir <dir>` — where to write `<DOC_ID>.json` files (defaults to `outputs/`)
- `--snapshot-dir <dir>` — also write an immutable copy for before/after comparison
- `--subset all|labelled|unlabelled` — run only a subset (defaults to `all`)
- `--model <model_id>` — override `GEMINI_MODEL`
- `--log-file <path>` — token usage JSONL file (defaults to a timestamped file in `logs/`)

### 4. Evaluation harness (10 labelled documents)

Run live extraction directly (no HTTP) and evaluate against ground truth:

```bash
python eval.py --verbose
```

Or evaluate already-saved outputs:

```bash
python eval.py --predictions-dir baselines/phase2_raw --verbose
python eval.py --predictions-dir outputs --verbose
```

Each run prints the per-field accuracy table, line-item sub-field accuracy, and review-flag stats, and saves a timestamped JSON summary to `eval_results/`.

### 5. Cost summary

```bash
python cost_summary.py logs/final_run_tokens.jsonl
```

Prints average input/output tokens per document, USD cost per document at paid-tier list price, first-attempt vs. retry cost split, and estimated cost per 1,000 invoices.

## Output Schema Extension (`needs_review`)

Every extraction result contains the 8 fields defined in `schema.json` plus two review metadata fields attached at the top level:

- `needs_review` (`boolean`): `true` if schema or arithmetic validation still failed after the single feedback retry.
- `review_reasons` (`list[string]`): specific human-readable reasons for any failed check (`[]` when valid).

These two fields are kept separate from the schema-checked `InvoiceExtraction` model. Calling `invoice_extractor.validator.strip_review_metadata(payload)` (or dropping `needs_review` and `review_reasons`) produces an object that validates directly against `schema.json`.

## Validation & Bad-Input Handling

- **Validation & retry once (D5, D6)**: Every model response is checked locally for strict JSON numeric types (rejecting string-encoded numbers), Pydantic `schema.json` conformance (including `is_invoice=false` all-null rules), and arithmetic consistency (`|sum(line_items.amount) + tax_total - total| <= 0.05`). On failure, the extractor retries **once** with the specific error message fed back; if it still fails, the output is returned with `needs_review: true` and `review_reasons`.
- **Extra bad-input case — API errors and timeouts (D7)**: Upstream Gemini calls use a bounded HTTP timeout (`API_TIMEOUT_SECONDS`, default `45s`) and retry transient errors (`429`, `500`, `502`, `503`, `504`, timeouts, connection errors) up to `API_MAX_RETRIES` times with exponential backoff (`1s`, `2s`, ...). If retries are exhausted, the pipeline returns a clear error instead of crashing (`502`/`504` JSON in FastAPI; exit code `3` in CLI; explicit error JSON in `batch_extract.py`).
- **File validation**: Missing files, unsupported extensions (anything other than `.pdf`, `.jpg`, `.jpeg`, `.png`), empty files, oversized uploads (`> MAX_UPLOAD_SIZE_MB`), and corrupt/unreadable PDFs or images return clean `400` JSON errors in FastAPI (with temporary upload files always deleted in `finally:`) and exit code `2` in CLI.
