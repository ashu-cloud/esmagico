# Invoice Extraction Pipeline — Evaluation & Cost Report

## Summary

- **Clean labelled set (`L01`–`L10`)**: **10/10 (100.0%)** exact documents and `66/66` line items (development-set result — prompt rules were written after inspecting all 20 sample documents).
- **Degraded robustness (`seed=42`, 5 degradations)**: **48/50 (96.0%)** exact documents and `329/330` line items across outage-free runs (`1` blur quantity misread, `1` physical edge-crop vendor error).
- **Review-flag check (`needs_review`, excluding `503` outages)**: **0 of 2 (0.0%)** real degraded extraction errors flagged (`0` false alarms), exposing the blind spot of arithmetic-only validation.
- **Cost per document**: **\$0.001292/doc** (`1,713.2` input / `311.2` output tokens avg; **\$1.29 per 1,000 invoices** at `gemini-3.5-flash-lite` paid list price).
- **Top fix**: add jittered backoff + fallback model (`gemini-3.1-flash-lite`) for `503` spikes, plus lightweight blur/border-crop and `quantity × unit_price` checks to catch non-arithmetic errors.

All numbers below come from real runs executed against the sample pack (`L01`–`L10` labelled, `U01`–`U10` unlabelled) on **2026-11-20** and manual visual verification of `U01`–`U10` on **2026-11-21**. No confidential vendor names, invoice numbers, or amounts from the sample pack are copied into this report.

---

## 1. Approach

### Model used and why
- **Model**: `gemini-3.5-flash-lite` via the Google Gemini API, sending rendered page images directly (`pymupdf` at 200 DPI for PDFs; Pillow with EXIF orientation correction for JPG/PNG; multi-page PDFs sent together in one request).
- **Model comparison on the 3 hardest labelled documents** (`L02.jpg` phone photo, `L09.pdf` scanned PDF, `L04.pdf` 2-page 24-row invoice) prior to selecting the default model:

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

### Prompting approach & interface architecture
- **Single general system prompt** (`invoice_extractor/prompt.py`) encoding the labelling conventions from the sample pack (`schema.json` fields, non-invoice all-null rule, row totals before tax, discounts as negative line items, sum of split tax lines, European number normalization, and ambiguous date inference from other dates on the same document).
- **No document-specific hints and no few-shot examples** from the labelled set.
- **Deterministic decoding**: `temperature=0.0` with Gemini structured JSON output (`response_mime_type="application/json"`, `response_json_schema=InvoiceExtraction.model_json_schema()`).
- **Architecture**: All extraction logic lives in `invoice_extractor/extractor.py` (framework-independent). FastAPI (`api.py` exposing `POST /extract`, `GET /health`, Swagger `/docs`, and a minimal side-by-side review frontend at `GET /` via `static/index.html`) is the primary interface; `extract.py` is a thin CLI wrapper (`python extract.py invoice.pdf > out.json`), and `batch_extract.py`, `eval.py`, and `robustness_eval.py` call `extract_invoice()` directly without going through HTTP.

### Validation and failure policy
1. **Strict numeric type check + local Pydantic validation** (`invoice_extractor/validator.py`): explicitly rejects string-encoded numbers (e.g., `"1,200.00"`) before Pydantic coercion, enforces `extra="forbid"`, validates `YYYY-MM-DD` and uppercase 3-letter ISO currency patterns, and enforces the `is_invoice=false` all-null rule.
2. **Arithmetic check**: for invoices (`is_invoice=true`), verifies `|sum(line_items.amount) + tax_total - total| <= 0.05` (absolute tolerance in the document's own currency to allow per-line tax rounding).
3. **Retry once, then flag**: on any schema or arithmetic failure, the pipeline retries **once** (`attempt=2`), feeding back the previous JSON and exact failed check message(s). If it still fails, it returns the payload with `needs_review: true` and `review_reasons: [...]` — never silently passing bad data.
4. **Schema-compatible output extension**: `needs_review` and `review_reasons` are attached at the top level and cleanly separable via `strip_review_metadata()`, leaving an 8-field object that validates directly against `schema.json`.

### Extra bad-input case chosen and how it is handled
- **Chosen extra case**: **Upstream API errors and timeouts**.
- Every Gemini API call uses a configurable HTTP timeout (`API_TIMEOUT_SECONDS=45`) and retries transient errors (`429`, `500`, `502`, `503`, `504`, `TimeoutError`, connection errors) up to `API_MAX_RETRIES` times (default `2`) with exponential backoff (`API_BACKOFF_BASE_SECONDS`, default `1.0`s), logging every attempt. Non-transient `4xx` errors fail fast without wasting retries.
- If retries are exhausted, the pipeline returns a clear, secret-free error without crashing: HTTP `504` (timeout) or `502` (provider error) JSON in FastAPI, exit code `3` in CLI, and an explicit error/flagged record in batch & eval runners.
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
3. **Short retry window (`API_MAX_RETRIES=2`, `API_BACKOFF_BASE_SECONDS=1.0`, ~3s total wait) during sustained `HTTP 503` spikes**:
   - *Expected*: 2 quick retries would ride out transient `503 UNAVAILABLE` responses during bulk robustness testing.
   - *What happened*: Under real Gemini capacity spikes, 6 of 50 degraded documents and 6 of 30 stability runs hit 3 consecutive `503`s within ~30s and failed.
   - *What I did instead*: Exposed `API_BACKOFF_BASE_SECONDS` alongside `API_MAX_RETRIES` as configuration and re-ran the affected subsets with `API_MAX_RETRIES=5` and `API_BACKOFF_BASE_SECONDS=2.0` (`2s, 4s, 8s, 16s, 32s` backoff), reducing degraded outage failures from `6` to `0` and stability outage failures from `6` to `2` without touching the prompt or validation logic (logged in Section 2's Tuning log).

---

## 2. Eval method

### Matching rules per field (`invoice_extractor/evaluator.py`)
- **`is_invoice`**: Strict boolean equality (`1 != True`).
- **`vendor`**: Lowercase, preserve `b.v.` as a single token before stripping punctuation, collapse whitespace, strip trailing legal suffixes (`pvt`, `ltd`, `limited`, `private`, `llp`, `llc`, `inc`, `gmbh`, `ag`, `b.v.`, `co`), and require exact equality (so `"Acme Pvt Ltd"` matches `"Acme"`, while `"Acme Traders"` does not).
- **`invoice_number`**: Trim whitespace and require case-insensitive exact match (slashes, hyphens, and prefixes must match).
- **`invoice_date`**: Exact string match on `YYYY-MM-DD`.
- **`currency`**: Exact match on uppercase 3-letter ISO code.
- **`tax_total` & `total`**: Numeric match within `±0.01` (`null` does NOT equal `0`). String-encoded numbers (such as `"1,200.00"`) are parsed for scoring if encountered, but explicitly counted as schema failures in validation stats rather than silent passes.
- **`line_items`**: Compared in printed order:
  - `description`: lowercased, punctuation-stripped, whitespace-collapsed; match if equal or `SequenceMatcher` ratio `>= 0.85`.
  - `quantity`: numeric match within `±0.001`; `null == null` only.
  - `amount`: numeric match within `±0.01`.
  - Row is correct only when all three sub-fields match. Row-level denominator uses `max(gt_rows, pred_rows)` per document so both missing and extra rows penalize row-level accuracy; document exact match requires equal row count and every row correct.
- **Non-invoices**: `is_invoice` must be `false` and every other field must be `null`.

### Commands to reproduce
```bash
# Score the preserved raw baseline (before validation/retry, no API calls)
python eval.py --predictions-dir baselines/phase2_raw --verbose

# Score the final validated outputs in outputs/ (no API calls)
python eval.py --predictions-dir outputs --verbose

# Run live extraction directly (no HTTP) on L01..L10 and evaluate
python eval.py --verbose

# Run reproducible degraded robustness eval (seed=42) + 3x clean stability eval
python robustness_eval.py --seed 42 --stability-runs 3
```

### Overfitting statement
- The prompt rules were written after the whole sample pack, including the unlabelled documents, had been inspected to anticipate patterns such as number formats and ambiguous dates. The clean 10/10 should therefore be read as a development-set result, not an independent test.
- No changes were made to the prompt (`invoice_extractor/prompt.py`), validator (`invoice_extractor/validator.py`), or evaluator (`invoice_extractor/evaluator.py`) after viewing any evaluation or robustness results. One transport-only configuration change was made after seeing `503` outages in the robustness and stability runs, disclosed below.

#### Tuning log
| # | Change | Why | Looked at which results first? | Effect on accuracy |
|---|---|---|---|---|
| 1 | Raised `API_MAX_RETRIES` from `2` to `5` and `API_BACKOFF_BASE_SECONDS` from `1.0` to `2.0` (environment config only) | Saw `503` outage failures in the robustness and stability runs (`6/50` degraded docs and `6/30` clean stability runs) | Robustness Run 1 & Stability Run 1 (`20261120T154656Z`) | Transport-only; no effect on extraction logic. Re-ran `jpeg_compression`, `partial_crop`, and 3× clean stability; outages dropped `6 → 0` (degraded) and `6 → 2` (stability), and the later run is the one reported in the combined 50-doc totals. |

*Disclosure note*: Raised `API_MAX_RETRIES` from 2 to 5 and `API_BACKOFF_BASE_SECONDS` from 1.0 to 2.0 after seeing `503` outage failures in the robustness and stability runs. Transport-only; no effect on extraction logic. The two affected degradation sets were re-run, and the later run is the one reported (both Run 1 and Run 2 files are preserved in `eval_results/`).

---

## 3. Accuracy & Robustness

### 3.1 Clean labelled set (`L01`–`L10`): Baseline vs. Final
Baseline (`baselines/phase2_raw`, `eval_results/eval_20261120T090504Z.json`) vs. Final (`outputs/`, `eval_results/eval_20261120T091209Z.json`):

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

- **Line-item sub-fields (both Baseline and Final)**: `description` 66/66 (100.0%), `quantity` 66/66 (100.0%), `amount` 66/66 (100.0%), predicted rows = 66, ground-truth rows = 66, string-number schema failures = 0.
- **Caveat**: 10 documents is a very small sample (one document = 10 percentage points), and as noted in Section 2, this is a development-set score rather than an independent holdout test.

### 3.2 Manual verification of the unlabelled set (`U01`–`U10`)
Each unlabelled document (`U01`–`U10`) was opened visually side-by-side against `outputs/U01.json`–`U10.json` and hand-checked across `is_invoice`, `vendor`, `invoice_number`, `invoice_date`, `currency`, `line_items`, `tax_total`, and `total`:

| Doc | Document Type / Notable Trait | `is_invoice` | Schema & Arith (`0.00`) | Manual Check (`vendor`, `invoice_number`, `invoice_date`, `currency`, items & totals) |
|---|---|---|---|---|
| `U01` | Digital PDF, 3 split tax lines | `true` | Pass | All fields match printed document (3 tax lines summed accurately) |
| `U02` | Tilted phone photo (JPG), `MM/DD/YYYY` | `true` | Pass | All fields match printed document (issue date chosen over due date & PO ref) |
| `U03` | Digital PDF, no tax line, ALL-CAPS header | `true` | Pass | All fields match printed document (`tax_total=0.0`; header casing preserved as printed) |
| `U04` | Digital PDF, cross-border buyer, zero tax | `true` | Pass | All fields match printed document (printed currency used, not buyer's country) |
| `U05` | 90°-rotated scanned PDF (no text layer) | `true` | Pass | All fields match printed document despite sideways scan |
| `U06` | Bank statement referencing invoice numbers | `false` | Pass | Correctly classified non-invoice; all other fields `null` |
| `U07` | Non-English (German) PDF, comma decimals | `true` | Pass | All fields match printed document (`DD.MM.YYYY` & European numbers normalized) |
| `U08` | European PDF (`B.V.`), 2 split VAT lines | `true` | Pass | All fields match printed document (both VAT lines summed; comma decimals normalized) |
| `U09` | Digital PDF, ambiguous date & surcharge row | `true` | Pass | All fields match printed document (date order resolved via due date; surcharge `quantity=null`) |
| `U10` | Cover letter mentioning an expected stipend | `false` | Pass | Correctly classified non-invoice; all other fields `null` |

### 3.3 Degraded-image robustness evaluation (`seed=42`, `L01`–`L10`)
To probe failure modes beyond the clean development set, we tested the untouched pipeline on 5 reproducible degradations (`generate_degraded.py --seed 42`, stored in git-ignored `degraded_samples/`) scored with the untouched main evaluator.

Two runs were executed with different transport retry configs (zero prompt/validator/evaluator changes):
- **Run 1** (`eval_results/robustness_20261120T154656Z.json`): `API_MAX_RETRIES=2`, `API_BACKOFF_BASE_SECONDS=1.0` (all 5 degradations; Gemini experienced intermittent `HTTP 503 UNAVAILABLE` spikes, `0` rate-limit `429`s).
- **Run 2** (`eval_results/robustness_20261120T162619Z.json`): `API_MAX_RETRIES=5`, `API_BACKOFF_BASE_SECONDS=2.0` (re-ran `jpeg_compression` and `partial_crop`, which had outages in Run 1).

| Degradation (`seed=42`) | Run & Retry Config | Outage-Exhausted (`503`) | End-to-End Doc Exact | End-to-End Row Acc | Conditional Doc Exact (API OK) | Conditional Row Acc (API OK) |
|---|---|---:|---|---|---|---:|
| `blur` (Gaussian `r=2.5`) | Run 1 (`retries=2, base=1s`) | 0 | 9 / 10 (90.0%) | 65 / 66 (98.5%) | **9 / 10 (90.0%)** | 65 / 66 (98.5%) |
| `rotation` (`±4°` to `±8°`) | Run 1 (`retries=2, base=1s`) | 0 | 10 / 10 (100.0%) | 66 / 66 (100.0%) | **10 / 10 (100.0%)** | 66 / 66 (100.0%) |
| `low_resolution` (`35%` scale) | Run 1 (`retries=2, base=1s`) | 0 | 10 / 10 (100.0%) | 66 / 66 (100.0%) | **10 / 10 (100.0%)** | 66 / 66 (100.0%) |
| `jpeg_compression` (`q=10`) | Run 1 (`retries=2, base=1s`) | 1 (`L05`) | 9 / 10 (90.0%) | 56 / 66 (84.8%) | **9 / 9 (100.0%)** | 56 / 56 (100.0%) |
| `jpeg_compression` (`q=10`) | Run 2 (`retries=5, base=2s`) | **0** | **10 / 10 (100.0%)** | **66 / 66 (100.0%)** | **10 / 10 (100.0%)** | 66 / 66 (100.0%) |
| `partial_crop` (`6%–10%` 2 edges) | Run 1 (`retries=2, base=1s`) | 5 (`L01, L02, L04, L06, L07`) | 4 / 10 (40.0%) | 24 / 66 (36.4%) | **4 / 5 (80.0%)** | 24 / 24 (100.0%) |
| `partial_crop` (`6%–10%` 2 edges) | Run 2 (`retries=5, base=2s`) | **0** | **9 / 10 (90.0%)** | **66 / 66 (100.0%)** | **9 / 10 (90.0%)** | 66 / 66 (100.0%) |

Across all 50 outage-free degraded evaluations (Run 1 `blur`/`rotation`/`low_resolution` + Run 2 `jpeg_compression`/`partial_crop`): **48 / 50 documents exact (96.0%)** and **329 / 330 line-item rows exact (99.7%)**.

### 3.4 Clean 3-run stability evaluation (`L01`–`L10` repeated 3×)

| Experiment | Retry Config | Per-Run End-to-End Exact | Total End-to-End | Outage-Exhausted (`503`) | Conditional Exact (API OK) | Field Diffs Across Successful Runs |
|---|---|---|---:|---:|---:|---:|
| Stability Run 1 (`eval_results/stability_20261120T154656Z.json`) | `retries=2, base=1s` | Run 1: 6/10 · Run 2: 10/10 · Run 3: 8/10 | 24 / 30 (80.0%) | 6 (`L01, L04, L05, L10` in R1; `L04, L07` in R3) | **24 / 24 (100.0%)** | **0** |
| Stability Run 2 (`eval_results/stability_20261120T162619Z.json`) | `retries=5, base=2s` | Run 1: 9/10 · Run 2: 10/10 · Run 3: 9/10 | 28 / 30 (93.3%) | 2 (`L04` in R1 and R3) | **28 / 28 (100.0%)** | **0** |

- **Stability finding**: At `temperature=0.0` with `response_json_schema`, model output was **100% byte-for-byte deterministic** — across all `52 / 52` successful API responses in the two 3-run stability experiments (including all 9 single-page documents succeeding 3/3 times in Stability Run 2), there were **0 field differences**. Every run-to-run difference came exclusively from upstream `HTTP 503 UNAVAILABLE` outages, with the largest 2-page 24-row PDF (`L04`) being the most susceptible under heavy server load.

---

## 4. Failure analysis

### Observed errors across clean + robustness + stability runs (grouped by cause)

| Run / Set | Doc | Field | Error Detail | `needs_review` | Cause Group |
|---|---|---|---|---|---|
| Clean `L01`–`L10` | *(None)* | — | 0 mismatches under evaluation rules | — | — |
| Degraded `blur` (`r=2.5`) | `L09` | `line_items[1].quantity` | Expected `18`, got `20.0` | `false` (missed) | Scan + blur digit misread |
| Degraded `partial_crop` (Run 1 & Run 2) | `L05` | `vendor` | Left `8.9%` edge crop physically removed the first character of the printed seller name; model returned the remaining clipped text | `false` (missed) | Physical crop / input damage |
| Degraded Run 1 (`jpeg_compression`, `partial_crop`) | `L05` (`jpeg`); `L01, L02, L04, L06, L07` (`crop`) | All 8 fields (`None`) | Exhausted 3 attempts on `HTTP 503 UNAVAILABLE` | `true` (flagged) | Upstream provider outage (`503`) |
| Stability Run 1 & Run 2 (clean 3×) | Run 1: `L01, L04, L05, L10`, `L04, L07` · Run 2: `L04` (×2) | All 8 fields (`None`) | Exhausted transport retries on `HTTP 503 UNAVAILABLE` (`0` `429` rate limits) | `true` (flagged) | Upstream provider outage (`503`) |

### Cause group breakdown

1. **Upstream provider capacity outages (`HTTP 503 UNAVAILABLE`)**
   - *Frequency*: 6/50 degraded docs and 6/30 stability runs under `retries=2, base=1s`; dropped to **0/20** degraded docs and **2/30** stability runs (both on 2-page `L04`) under `retries=5, base=2s`.
   - *Root cause*: Transient high demand on Google's `gemini-3.5-flash-lite` endpoint (confirmed `0` HTTP `429` rate-limit errors in all token logs). Larger 2-page inputs (`L04`) experienced higher `503` failure probability.
   - *Model error or operational?*: Operational infrastructure failure — cleanly caught and flagged `needs_review: true` rather than crashing or silently returning corrupt fields.

2. **Scan + blur digit misread (`blur/L09`)**
   - *Frequency*: 1 error across 50 degraded documents (`blur/L09`, `line_items[1].quantity`: `18` vs. `20.0`).
   - *Root cause*: `L09` is already a rasterized scan with no text layer; adding `radius=2.5` Gaussian blur smeared small digits in the narrow quantity column while leaving the wider amount column legible.
   - *Model error or eval artefact?*: Genuine model vision error — and critically, it passed validation unflagged because `amount`, `tax_total`, and `total` were all read accurately (`diff = 0.00`).

3. **Physical input loss from edge cropping (`partial_crop/L05`)**
   - *Frequency*: 1 error (reproduced identically in Run 1 and Run 2 on `partial_crop/L05`, `vendor`).
   - *Root cause*: The seeded `8.9%` left-edge crop literally sliced off the first letter of the vendor name at the top-left margin. The model transcribed the visible pixels faithfully, so the output could not match ground truth without an external vendor master list.
   - *Model error or eval artefact?*: Physical input-quality failure (unrecoverable from pixels alone; also unflagged because a clipped string still passes schema and arithmetic checks).

4. **Raw casing variance (non-error under eval rules)**
   - *Frequency*: 3 clean labelled docs (`L05`, `L07`, `L09`) and 1 unlabelled doc (`U03`) had all-caps printed vendor headers vs. title-case ground truth (7/10 clean labelled docs matched byte-for-byte). Properly normalized by the evaluator's case-insensitive vendor rule.

### What I would fix first (ranked by errors removed per effort)
1. **Increase default backoff window + add jitter & secondary model fallback (`gemini-3.1-flash-lite`)**: Config-only tuning (`retries=5, base=2s`) already eliminated 100% of degraded outages (`6 → 0`) and 67% of stability outages (`6 → 2`). Adding jitter and falling back to `gemini-3.1-flash-lite` after persistent `503`s removes the remaining `L04` outage risk with minimal code.
2. **Pre-extraction image-quality gate (blur & border-clipping check)**: A few lines of Pillow/NumPy checking sharpness (Laplacian variance) and non-white ink touching the image border would catch both `blur/L09`-class blurry scans and `partial_crop/L05`-class clipped headers and raise `needs_review: true` immediately.
3. **Quantity × unit-price cross-check (when unit price is printed)**: Adding an optional internal consistency check catches silent `quantity` misreads like `blur/L09` (`18 → 20.0`) that row-amount summation cannot detect.
4. **Expand the labelled benchmark (50–100+ real-world noisy invoices)**: Needed before tuning prompts or thresholds further so improvements generalize beyond a 20-document development pack.

---

## 5. Validation and review flag

### Clean 20-document batch run (`outputs/`, `logs/final_run_tokens.jsonl`)
- Documents failing a check on Attempt 1: **0 / 20**
- Documents fixed by validation retry: **0**
- Documents ending `needs_review: true`: **0 / 20** (all 17 invoices balanced to `0.00`; all 3 non-invoices passed the all-null rule).

### Bonus review-flag check — **excluding outage failures from precision & recall**
Reporting `needs_review` precision/recall with `HTTP 503` outages included would overstate the validator's accuracy (`6/8` wrong docs flagged in Run 1 were all `503` crashes, not caught extraction mistakes). Measuring **strictly on documents that received a successful API response** gives the honest signal:

| Evaluation Set (Non-Outage Docs Only) | Docs Evaluated (API OK) | Actual Wrong Docs | Flagged (`needs_review=true`) | Wrong Docs Flagged (TP) | Unflagged Wrong (FN / Misses) | Flagged Correct (FP / False Alarms) | Flag Recall (excl. outages) | Flag Precision (excl. outages) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Clean labelled (`outputs/`) | 10 | 0 | 0 | 0 | 0 | 0 | N/A (0 wrong) | N/A (0 flagged) |
| Degraded Run 1 (all 5 types, excl. 6 outages) | 44 | 2 | 0 | 0 | 2 (`blur/L09`, `crop/L05`) | 0 | **0 / 2 (0.0%)** | N/A (0 flagged) |
| Degraded Run 2 (`jpeg` + `crop`, excl. 0 outages) | 20 | 1 | 0 | 0 | 1 (`crop/L05`) | 0 | **0 / 1 (0.0%)** | N/A (0 flagged) |
| **Combined 50 degraded (outage-free)** | **50** | **2** | **0** | **0** | **2** | **0** | **0 / 2 (0.0%)** | **N/A (0 flagged)** |

### Controlled fault-injection verification (synthetic numeric & schema errors)
- *Persistent broken total* (off by `10.00` > `0.05` tolerance): triggered Attempt 2 with the exact arithmetic discrepancy fed back, logged both `attempt=1` and `attempt=2`, and returned `needs_review: true` with the failure reason in `review_reasons`.
- *Transient broken total fixed on retry*: triggered Attempt 2 and returned `needs_review: false` once corrected.
- *String-encoded number (`"165.00"`)*: rejected before Pydantic coercion, retried, and flagged `needs_review: true`.

### Honest conclusion on the review flag
Arithmetic + schema validation achieves **0 false alarms** and reliably catches broken totals, missed/extra rows, sign errors, schema violations, and upstream `503` outages — **but achieved 0% recall (0/2) on the two real degraded-image extraction errors** (`blur/L09` quantity misread and `partial_crop/L05` clipped vendor name) because neither error breaks `sum(line_items.amount) + tax_total == total`. Production flagging needs image-quality and semantic checks alongside arithmetic validation.

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
- **Cost of retries (share of total)**: **\$0.000000 (0.0% of total)** *(in production, if 5%–10% of real-world invoices trigger a single validation retry, effective cost rises roughly 5%–10% to ~\$0.00136–\$0.00142/doc; failed `HTTP 503` calls billed `0` tokens)*
- **Method**: Exact `prompt_token_count`, `candidates_token_count`, and `thoughts_token_count` read from Gemini API `usage_metadata` for every API call (including retries), summed across all calls, multiplied by the paid-tier list price per token, and averaged over all 20 documents. Runs used Google AI Studio's free tier (\$0 out-of-pocket), while all USD figures above are computed from Google's official paid-tier list price. Production use on sensitive invoices should enable paid billing for non-training data terms.
- **Estimated cost to process 1,000 invoices**: **\$1.29** (at 0% retry rate; ~\$1.36–\$1.42 assuming 5%–10% validation retries).

---

## 7. Limitations and next steps

### What is not handled
- **No independent holdout set**: The prompt rules were designed after inspecting all 20 sample documents, and formal ground truth is only available on 10 of them (`L01`–`L10`). For `U01`–`U10`, all 10 outputs were manually hand-verified against their source PDFs/JPG across `is_invoice`, `vendor`, `invoice_number`, `invoice_date`, `currency`, `line_items`, `tax_total`, and `total` (all 10 matched by eye, including non-invoices `U06` and `U10`), though official reviewer ground-truth JSONs remain withheld.
- **Non-arithmetic silent errors**: As demonstrated by `blur/L09` (quantity) and `partial_crop/L05` (clipped vendor), arithmetic validation alone cannot flag errors that leave row amounts and totals intact.
- **Upstream `503` availability on large multi-page PDFs**: Under heavy Gemini demand, 2-page `L04` still exhausted 6 attempts in 2 of 3 stability runs without a secondary fallback model.
- **Additional file formats & edge cases**: Password-protected/encrypted PDFs, multi-document PDFs (multiple invoices merged into one file), HEIC/TIFF/WebP/DOCX files, and credit notes with negative totals are not explicitly handled.
- **Production API hardening**: FastAPI currently has no authentication, rate limiting, CORS configuration, or async job queue for batch uploads.

### What I would do with more time
1. Add jittered backoff plus automatic fallback to `gemini-3.1-flash-lite` (\$0.25/\$1.50 per 1M) when `gemini-3.5-flash-lite` returns sustained `503`s.
2. Add lightweight image-quality pre-checks (sharpness/blur score + border-crop detection) and optional `quantity × unit_price` cross-checks to raise review-flag recall on non-arithmetic errors.
3. Expand the labelled benchmark to 50–100+ unseen real-world invoices (crumpled receipts, multi-currency credit notes, 5+ page tables) and run a side-by-side accuracy/cost comparison with `gemini-3.1-flash-lite` and an open-weight vision model.
4. Add authentication, rate limiting, and background job queues for multi-file batch uploads via the web UI.
