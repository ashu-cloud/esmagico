# Invoice Extraction Pipeline — Evaluation & Cost Report

All numbers below come from real runs executed against the sample pack (`L01`–`L10` labelled, `U01`–`U10` unlabelled) on **2026-11-20**. No confidential document text, vendor names, or amounts from the sample pack are copied into this report.

---

## 1. Approach

### Model used and why (Decisions D1, D2)
- **Model**: `gemini-3.5-flash-lite` via Google Gemini API, sending rendered page images directly (`pymupdf` at 200 DPI for PDFs; Pillow with EXIF orientation correction for JPG/PNG; multi-page PDFs sent together in one request).
- **Model comparison on the 3 hard documents** (`L02.jpg` phone photo, `L09.pdf` scanned PDF, `L04.pdf` 2-page 24-row invoice) prior to selecting the default model:

| Candidate Model | Status on 3 Hard Docs | Avg Latency | Output Tokens (3 docs, incl. thoughts) | Paid List Price (In / Out per 1M, checked 2026-11-20) |
|---|---|---:|---:|---|
| `gemini-2.5-flash-lite` | `404 NOT_FOUND` (retired for new users) | — | — | Retired (Google recommends `gemini-3.5-flash-lite`) |
| `gemini-3.1-flash-lite` | All 3 succeeded, 0 arithmetic diff, 100% row/amount/qty match | 5.7s | 1,921 (0 thoughts) | \$0.25 / \$1.50 |
| **`gemini-3.5-flash-lite`** *(chosen)* | **All 3 succeeded, 0 arithmetic diff, 100% row/amount/qty match** | **4.1s** | **1,956 (0 thoughts)** | **\$0.30 / \$2.50** |
| `gemini-3.5-flash` | All 3 succeeded, identical accuracy | 27.1s | 3,937 (2,747 thoughts) | \$1.50 / \$7.50 (deprecated) |
| `gemini-3.6-flash` | `504 DEADLINE_EXCEEDED` on `L02.jpg`; 2/3 ok | >34s | 3,435 on 2 docs (2,371 thoughts) | \$1.50 / \$7.50 |
| `gemini-3.8-flash` | `503 UNAVAILABLE` on `L02.jpg`; hung >6m on `L09.pdf` | unavailable | — | \$0.75 / \$3.75 (introductory) |

- **Why `gemini-3.5-flash-lite`**: Matched the accuracy of larger Flash models across all three hard documents while running ~6x faster (4.1s vs. 27.1s), emitting zero thinking tokens by default, costing ~5x–6x less at list price, and serving as Google's current recommended Flash-Lite model.

### Why not OCR plus text
- 5 of the 20 sample documents have no PDF text layer (3 scanned PDFs and 2 tilted phone photos). A text-layer extractor alone fails on 25% of the dataset.
- Running a separate OCR step before an LLM introduces a second failure source (broken table column alignment, dropped negative signs, misread European decimal commas) and extra system complexity.
- One unified page-image path is simpler to build, test, and explain, and the measured token cost is only **\$0.00129 per document**.

### Prompting approach & interface architecture (D3, D8, D9)
- **Single general system prompt** (`invoice_extractor/prompt.py`) containing the `schema.json` field rules, non-invoice classification rules (returning `is_invoice: false` and all other fields `null`), European number normalization (decimal comma / dot thousands separator), and ambiguous date resolution using other dates on the same page.
- **No document-specific hints and no few-shot examples** from the labelled set.
- **Deterministic decoding**: `temperature=0.0` with Gemini structured JSON output (`response_mime_type="application/json"`, `response_json_schema=InvoiceExtraction.model_json_schema()`).
- **Architecture (D9, updated 2026-11-20)**: All extraction logic lives in `invoice_extractor/extractor.py` (framework-independent). FastAPI (`api.py` with `POST /extract`, `GET /health`, and Swagger `/docs`) is the primary interface, `extract.py` is a thin CLI wrapper, and `batch_extract.py` / `eval.py` call `extract_invoice()` directly without going through HTTP.

### Validation and failure policy (D4, D5, D6)
1. **Strict numeric type check + local Pydantic validation** (`invoice_extractor/validator.py`): explicitly rejects string-encoded numbers (e.g., `"1,200.00"`) before Pydantic coercion, enforces `extra="forbid"`, validates `YYYY-MM-DD` and uppercase 3-letter ISO currency patterns, and enforces the `is_invoice=false` all-null rule.
2. **Arithmetic check**: for invoices (`is_invoice=true`), verifies `|sum(line_items.amount) + tax_total - total| <= 0.05` (absolute tolerance in document currency, D6).
3. **Retry once, then flag**: on any schema or arithmetic failure, the pipeline retries **once** (`attempt=2`), feeding back the previous JSON and exact failed check message(s). If it still fails, it returns the payload with `needs_review: true` and `review_reasons: [...]` — never silently passing bad data.
4. **Schema-compatible output extension**: `needs_review` and `review_reasons` are attached at the top level and cleanly separable via `strip_review_metadata()`, leaving an 8-field object that validates directly against `schema.json`.

### Extra bad-input case chosen and how it is handled (D7)
- **Chosen extra case**: **Upstream API errors and timeouts**.
- Every Gemini API call uses an explicit HTTP timeout (`API_TIMEOUT_SECONDS=45`). Transient errors (`429`, `500`, `502`, `503`, `504`, `TimeoutError`, connection errors) are retried up to `API_MAX_RETRIES=2` times with exponential backoff (`1s`, `2s`), logging every attempt. Non-transient `4xx` errors fail fast without wasting retries.
- If retries are exhausted, the pipeline returns a clear, secret-free error without crashing: HTTP `504` (timeout) or `502` (provider error) JSON in FastAPI, exit code `3` in CLI, and an explicit error JSON record in `batch_extract.py`.
- Standard bad inputs (missing file, unsupported extension, empty file, upload over `MAX_UPLOAD_SIZE_MB`, corrupt PDF/image) return HTTP `400` in FastAPI (with temporary upload files always deleted in `finally:`) and exit code `2` in CLI.

### Things I tried that did not work
1. **Passing Pydantic's `InvoiceExtraction` directly to Gemini's `response_schema`**:
   - *Expected*: The `google-genai` SDK would convert `InvoiceExtraction` directly into Gemini's structured-output schema.
   - *What happened*: Every API call failed with `400 INVALID_ARGUMENT: Unknown name "additional_properties" at 'generation_config.response_schema'` because `InvoiceExtraction` uses `extra="forbid"` (mirroring `schema.json`'s `additionalProperties: false`), which Gemini's legacy OpenAPI `response_schema` rejects.
   - *What I did instead*: Switched to `response_json_schema=InvoiceExtraction.model_json_schema()` (standard JSON Schema support), keeping `extra="forbid"` for both API-side schema adherence and local Pydantic validation.
2. **Opening uploaded PDFs directly by path in PyMuPDF on Windows**:
   - *Expected*: `with pymupdf.open(tmp_path)` would close cleanly on error so `api.py`'s `finally:` block could delete the temp file.
   - *What happened*: When testing a corrupt PDF (`corrupt.pdf`), PyMuPDF raised `FileDataError` while keeping a C-level stream handle referenced in the exception traceback, causing `PermissionError: [WinError 32]` during temp-file cleanup and returning a `500` instead of `400`.
   - *What I did instead*: Read file bytes into memory first (`path.read_bytes()`) and pass them to `pymupdf.open(stream=..., filetype="pdf")` (and `Image.open(io.BytesIO(...))`), eliminating OS file locks so `finally:` cleanup always succeeds.

---

## 2. Eval method

### Matching rules per field (summarising `04_EVAL_RULES.md`)
- **`is_invoice`**: Strict boolean equality (`1 != True`).
- **`vendor`**: Lowercase, keep `b.v.` intact before stripping punctuation, collapse whitespace, strip trailing legal suffixes (`pvt`, `ltd`, `limited`, `private`, `llp`, `llc`, `inc`, `gmbh`, `ag`, `b.v.`, `co`), and require exact equality.
- **`invoice_number`**: Trim whitespace and require case-insensitive exact match (slashes, hyphens, and prefixes must match).
- **`invoice_date`**: Exact string match on `YYYY-MM-DD`.
- **`currency`**: Exact match on uppercase 3-letter ISO code.
- **`tax_total` & `total`**: Numeric match within `±0.01` (`null` does NOT equal `0`). String-encoded numbers are parsed for scoring if encountered, but explicitly counted as schema failures in validation stats.
- **`line_items`**: Compared in printed order:
  - `description`: lowercased, punctuation-stripped, whitespace-collapsed; match if equal or `SequenceMatcher` ratio `>= 0.85`.
  - `quantity`: numeric match within `±0.001`; `null == null` only.
  - `amount`: numeric match within `±0.01`.
  - Row is correct only when all three sub-fields match. Row-level denominator uses `max(gt_rows, pred_rows)` per document so both missing and extra rows penalize row-level accuracy; document exact match requires equal row count and every row correct.
- **Non-invoices**: `is_invoice` must be `false` and every other field must be `null`.

### Commands to reproduce
```bash
# Score the preserved Phase 2 raw baseline (no API calls)
python eval.py --predictions-dir baselines/phase2_raw --verbose

# Score the final validated outputs in outputs/ (no API calls)
python eval.py --predictions-dir outputs --verbose

# Or run live extraction directly (no HTTP) on L01..L10 and evaluate
python eval.py --verbose
```

### Overfitting statement
- **Zero prompt or rule tuning was performed using the labelled documents.** The system prompt (`invoice_extractor/prompt.py`) and validation rules (`invoice_extractor/validator.py`) were written once from `01_SPEC.md` and `02_DECISIONS.md` before running the Phase 3 evaluation, and were not modified after seeing labelled scores.

#### Tuning log
| # | Change | Why | Looked at which labelled results first? | Effect on accuracy |
|---|---|---|---|---|
| — | *(None — no post-eval prompt or rule tuning performed)* | Baseline already scored 10/10 under `04_EVAL_RULES.md` | None | N/A |

---

## 3. Accuracy

Baseline (`baselines/phase2_raw`, saved in `eval_results/eval_20261120T090504Z.json`, before validation/retry) vs. Final (`outputs/`, saved in `eval_results/eval_20261120T091209Z.json`, with validation/retry and timeout/backoff enabled):

| Field | Baseline | Final |
|---|---|---|
| is_invoice | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| vendor | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| invoice_number | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| invoice_date | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| currency | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| line_items (row-level) | 66 / 66 (100.0%) | 66 / 66 (100.0%) |
| line_items (document exact) | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| tax_total | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| total | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| All fields correct (document) | 10 / 10 (100.0%) | 10 / 10 (100.0%) |

- **Line-item sub-field breakdown (both Baseline and Final)**: `description` 66/66 (100.0%), `quantity` 66/66 (100.0%), `amount` 66/66 (100.0%), predicted rows = 66, ground-truth rows = 66.
- **String-number schema failures**: 0.
- **Caveat**: 10 documents is a very small sample — one document equals 10 percentage points. A 100% score on 10 synthetic documents should **not** be mistaken for 100% reliability on arbitrary real-world invoices.

---

## 4. Failure analysis

### Mismatches on the 10 labelled documents
| Doc | Field | Expected | Got | Cause group |
|---|---|---|---|---|
| *(None)* | — | — | — | 0 mismatches under `04_EVAL_RULES.md` |

### Sub-eval observations & operational failures seen during development
Although all 10 labelled documents passed `04_EVAL_RULES.md` with 0 field errors (and all 17 invoices across `L01`–`U10` balanced with `0.00` arithmetic difference, while all 3 non-invoices `L08`, `U06`, `U10` returned `is_invoice: false` and all nulls), auditing raw strings and model runs surfaced three real patterns worth noting:

1. **Raw casing differences on `vendor` (3 docs: `L05`, `L07`, `L09`)**:
   - *Observation*: 7 of 10 labelled documents matched ground truth byte-for-byte; on `L05`, `L07`, and `L09`, every field and line item matched byte-for-byte except upper/lower casing on `vendor` (e.g., all-caps header printed on the page vs. title-case ground truth).
   - *Classification*: Formatting variance, not a semantic error; properly handled by `04_EVAL_RULES.md`'s case-insensitive vendor normalization.
2. **Upstream availability & latency spikes during model selection (`gemini-3.8-flash`, `gemini-3.6-flash`)**:
   - *Observation*: `gemini-3.8-flash` returned `503 UNAVAILABLE` and then hung >6 minutes without an HTTP timeout; `gemini-3.6-flash` hit `504 DEADLINE_EXCEEDED` at 45s on `L02.jpg`.
   - *Root cause*: High demand + thinking-token generation latency on heavier Flash variants, compounded initially by missing HTTP timeouts.
   - *Fix applied*: Selected `gemini-3.5-flash-lite` (zero thinking tokens, ~3–4s/doc) and added explicit `API_TIMEOUT_SECONDS=45` + exponential backoff (`D7`).
3. **Likely failure modes on larger real-world datasets (what I would harden first, ranked by impact/effort)**:
   1. **Expand the labelled test set (highest priority)**: 10 clean/moderate synthetic documents cannot surface long-tail real-world errors (blurry photos, handwritten totals, multi-currency credit notes, 5+ page tables).
   2. **Non-numeric cross-checks**: Add lightweight deterministic checks (e.g., flag implausible invoice dates, empty vendor strings, or unrecognized currency/country mismatches) because arithmetic validation cannot catch a misread `invoice_date` or `invoice_number`.
   3. **Image pre-processing for very low-contrast / rotated scans**: Add optional deskew / higher-DPI retry if Attempt 1 fails arithmetic on a scanned PDF or phone photo.

---

## 5. Validation and review flag

- **Final 20-document batch run (`outputs/`, `logs/final_run_tokens.jsonl`)**:
  - Documents that failed a check on Attempt 1: **0 / 20**
  - Documents fixed by retry: **0**
  - Documents ending with `needs_review: true`: **0 / 20** (all 17 invoices had `0.00` arithmetic discrepancy; all 3 non-invoices passed the all-null rule).
- **Bonus review-flag check (on the 10 labelled documents)**:
  - Wrong documents flagged: **0 / 0**
  - Wrong documents unflagged (misses): **0**
  - Flagged documents that were actually right (false alarms): **0**
- **Controlled fault-injection verification (Phase 4 & 5 tests)**:
  - *Persistent broken total* (off by `10.00` > `0.05` tolerance): triggered Attempt 2 with the exact arithmetic discrepancy fed back, logged both `attempt=1` and `attempt=2`, and returned `needs_review: true` with the failure reason in `review_reasons`.
  - *Transient broken total fixed on retry*: triggered Attempt 2 and returned `needs_review: false` once the corrected total balanced.
  - *String-encoded number (`"165.00"`)*: rejected before Pydantic coercion, retried, and flagged `needs_review: true`.
- **Honest limitation of the review flag**: On a harder dataset, the arithmetic check has high precision for catching missed rows, sign errors on discounts, and misread amounts/taxes, **but zero recall for non-numeric errors** (such as a wrong `invoice_date`, `vendor`, or `invoice_number`).

---

## 6. Cost per document

Measured from the final 20-document run ([logs/final_run_tokens.jsonl](logs/final_run_tokens.jsonl)) using `python cost_summary.py logs/final_run_tokens.jsonl`:

- **Model**: `gemini-3.5-flash-lite`
- **Paid-tier list price (checked 2026-11-20, source: `ai.google.dev/gemini-api/docs/pricing`)**:
  - Input: **\$0.30 per 1M tokens**
  - Output (including thinking tokens): **\$2.50 per 1M tokens**
- **Total tokens across all 20 documents**: `34,264` input / `6,225` output (`0` thinking tokens; `20` first-attempt calls, `0` retries, `0` errors)
- **Average input tokens per document**: **1,713.2**
- **Average output tokens per document**: **311.2**
- **Average cost per document (USD)**: **\$0.001292** (~0.13 cents per document; total run cost for 20 docs = **\$0.025842**)
- **Cost of retries (share of total)**: **\$0.000000 (0.0% of total)** *(note: in production, if 5%–10% of real-world invoices trigger a single validation retry, effective cost rises by roughly 5%–10% to ~\$0.00136–\$0.00142/doc)*
- **Method**: Exact `prompt_token_count`, `candidates_token_count`, and `thoughts_token_count` read from Gemini API `usage_metadata` for every API call (including retries), summed across all calls, multiplied by the paid-tier list price per token, and averaged over all 20 documents. *(Note: runs executed via Google AI Studio free tier incur \$0 out-of-pocket charges; all USD figures above are computed from Google's official paid-tier list price. Confirm billing/data-handling tier in Google AI Studio if strict non-training data terms are required for production documents.)*
- **Estimated cost to process 1,000 invoices**: **\$1.29** (at 0% retry rate; ~\$1.36–\$1.42 assuming 5%–10% validation retries).

---

## 7. Limitations and next steps

### What is not handled
- **Unlabelled ground truth**: `U01`–`U10` have no ground-truth labels; while all 10 passed schema, non-invoice, and exact `0.00` arithmetic validation, non-numeric fields (`vendor`, `invoice_number`, `invoice_date`, `currency`, `description`) on `U01`–`U10` were not verified against ground truth.
- **Small labelled evaluation set**: Only 10 labelled documents exist, so statistical confidence is low and subtle real-world edge cases are under-represented.
- **Non-numeric silent errors**: Arithmetic validation cannot detect wrong dates, wrong invoice numbers, or subtle vendor-name misreads.
- **Additional file formats & edge cases**: Password-protected/encrypted PDFs, multi-document PDFs (multiple invoices merged into one file), HEIC/TIFF/WebP/DOCX files, and credit notes with negative totals are not explicitly handled.
- **Production API hardening**: FastAPI currently has no authentication, rate limiting, CORS configuration, or async job queue for very large multi-page PDFs.

### What I would do with more time
1. Build a larger, noisier labelled benchmark (50–100+ invoices including crumpled receipts, skewed scans, multi-currency credit notes, and multi-invoice PDFs) to stress-test failure modes and measure real review-flag recall/precision.
2. Add semantic sanity checks (e.g., date range checks, non-empty vendor/invoice_number checks on invoices, duplicate invoice detection) to improve review-flag coverage beyond arithmetic errors.
3. Run an optional side-by-side comparison against `gemini-3.1-flash-lite` (\$0.25/\$1.50 per 1M, ~30% cheaper) and an open-weight vision model (e.g., Groq) on the expanded benchmark.
4. Add a lightweight web UI (Bonus #2) calling `POST /extract` to display the uploaded invoice alongside extracted fields and `needs_review` warnings.
