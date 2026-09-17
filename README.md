# vlm-receipts

Benchmark harness for answering one question:

> **Which small vision-language model can parse a receipt photo, in a browser tab, on WebGPU, well
> enough to be trusted inside a personal-finance PWA?**

Every measurement is taken by running the model **inside a real Chrome tab**, driven by Playwright.
Nothing is estimated or extrapolated from server-side numbers, because the constraints that matter
here — VRAM, quantization behaviour, download size, in-tab latency — only exist in the browser.

---

## Quick start

```bash
npm install
npm run probe:gpu        # verify a real WebGPU adapter exists
npm run fixtures         # download receipt fixtures + ground truth
npm run serve            # harness dev server (watcher disabled for stable timing)

# in a second shell
npm test                 # scorer unit tests
npm run bench            # run the benchmark
npm run report           # markdown report from results/*.json
npm run charts           # SVG charts only
npm run dashboard        # single-page dashboard (charts + tables + findings)
npm run dashboard:serve  # serve it at http://127.0.0.1:5180/
```

`npm run serve` must be running before any `scripts/run-benchmark.mjs` invocation; the runner drives
a browser against `http://127.0.0.1:5179/`.

## Dashboard

```bash
npm run dashboard -- results/bench-final.rescored.json   # writes results/dashboard.html
npm run dashboard:serve                                  # open http://127.0.0.1:5180/
```

One self-contained HTML file with **everything in a single view**: KPI header, all five charts, the
overall and per-field result tables, the findings, PWA implications, and method/provenance. Charts are
inlined as SVG and the findings prose is pulled from `REPORT.md` at build time, so the dashboard and
the report cannot drift apart and the file works offline with no CDN or build step.

Add `--png` to any chart or dashboard command to also rasterize it. Worth doing after editing chart
code: hand-rolled SVG is easy to get subtly wrong, and overlapping or clipped labels are invisible if
you only read the generated markup.

## Charts

`npm run charts -- results/bench-final.rescored.json` writes the individual SVGs plus a chart-only
index to `results/charts/`. Use this when you want to embed a single chart; use `npm run dashboard`
when you want the whole result in one page.

| Chart | What it answers |
|---|---|
| `accuracy.svg` | Headline ranking, with JSON-validity and exact-match rates below each bar. |
| `field-heatmap.svg` | **The chart to choose a model by** — which fields each model actually fails on. |
| `tradeoff.svg` | Accuracy vs. latency; bubble area is download size. |
| `download-cost.svg` | Accuracy vs. download size, including MB-per-accuracy-point. |
| `parse-strategy.svg` | How much output needed recovering, and how much was unrecoverable. |

Both renderers share `renderCharts()` in `scripts/charts.mjs`, so the charts cannot differ between
the two outputs.

---

## What it measures

Each model is graded on a **strict transaction contract**, the shape a personal-finance app needs in
order to create a transaction without hand-correction:

```json
{
  "merchant": "GARCIA SUPERMARKET",
  "date": "2020-07-23",
  "total": 184.89,
  "currency": "GBP",
  "tax": 30.81,
  "line_items": [{ "description": "Coffee", "qty": 1, "unit_price": 3.5, "amount": 3.5 }]
}
```

Headline accuracy averages the four fields that actually create a transaction — `merchant`, `date`,
`total`, `currency`. `tax` and `line_items` are reported per-field but excluded from the headline,
because requiring itemisation would penalise models on a task the app can live without.

Scoring is deliberately generous where a human would be: near-miss merchant names earn partial
credit, totals within a cent or two are accepted, and partially-correct dates score proportionally.
It is strict where correctness is binary: a wrong currency is wrong.

### Dataset

`albertobarnabo/synthetic-receipts-ocr` (Apache-2.0) — 32,000 synthetic thermal receipts across
UK/US/DE/FR/IT locales, with machine-exact ground truth. Its `eval` split is deliberately de-leaked
by the dataset authors (held-out fonts, vocabulary, and merchant names), so scores are not inflated
by memorisation.

The **`image_photo` variant is used, not the clean render** — each fixture carries perspective,
lighting, shadow, noise, blur, and JPEG-artifact degradations, because a phone camera never sees a
clean render.

Two honest limits of this corpus: it is entirely synthetic (monospace thermal, Latin script, no
handwriting or crumpled paper), and the dataset authors explicitly leave real-photo transfer
unmeasured.

---

## Architecture

```
scripts/
  serve.mjs                    dev server with the file watcher disabled
  run-benchmark.mjs            the runner: builds fixtures, drives Chrome, scores in Node
  report.mjs                   renders results/*.json into a markdown report
  charts.mjs                   renders results/*.json into SVG charts (exports renderCharts)
  dashboard.mjs                single-page dashboard: charts + tables + findings + provenance
  serve-dashboard.mjs          serves the dashboard at http://127.0.0.1:5180/
  rescore.mjs                  re-grades stored output with the current scorer
  compatibility-check.mjs      fast gate: does each candidate load, run, and stay coherent?
  merge-results.mjs            combines per-model result files into one ranking
  screenshot.mjs               screenshots a tall page in slices, for visual verification
  repair-encoding.mjs          repairs Windows CP-1252 mojibake in source files
  extract-parquet-fixtures.mjs fetches fixtures + labels via parquet range requests
  fetch-fixtures.mjs           alternative fixture source (datasets-server /rows)
  probe-webgpu.mjs             standalone WebGPU capability probe
  probe-model.mjs              probe one (model, dtype) with coherence + schema-recital gates
  model-footprint.mjs          true download sizes, including external .onnx_data sidecars
  trace-lfm2.mjs               traces the LFM2.5-VL processor failure in isolation
  main-thread-lfm2.mjs         reproduces the LFM2.5-VL failure on the main thread
  inspect-processor.mjs        dumps an unfamiliar VLM processor's class and config
  discover-onnx-models.mjs     finds ONNX VLM repos and real quantized sizes
  check-extra-models.mjs       verifies candidate repos publish usable ONNX weights
  sweep-quantizations.mjs      tests each dtype per model (see "quantization" below)
  test-quantizations.mjs       per-dtype comparison for a single repo
  test-constrained.mjs         A/B harness for decoding configurations
  inspect-outputs.mjs          raw output per prompt variant, next to ground truth
  dump-output.mjs              single (model, receipt) raw dump for debugging
  debug-smolvlm.mjs            chat-template / image-pipeline diagnostics
  smoke-test.mjs               minimal one-model, one-image end-to-end check
src/
  models.js                    candidate registry + documented exclusions
  prompts.js                   benchmarked prompt variants
  receipt-schema.js            the JSON schema contract
  schema.js                    extraction + scoring + aggregation (pure, unit-tested)
  vlm-worker.js                in-browser inference worker (WebGPU)
  bench-api.js                 window.__VLM_BENCH__ API used by the runner
  debug-api.js                 inspection entry point for diagnostics
tests/schema.test.mjs          44 tests over the grader
docs/research/                 background research that informed the design (inputs, not results)
```

Background research — browser/WebGPU capability surveys, the model shortlist, and the dataset
comparison — lives in `docs/research/`. Treat those as inputs: where they disagree with the measured
results, the results win, and `docs/research/README.md` records the specific disagreements.

Three design decisions exist to protect measurement validity:

1. **Scoring happens in Node, never in the page.** A model cannot influence how it is graded.
2. **One browser context per model, closed afterwards.** A leaky model cannot contaminate the next
   model's VRAM measurements.
3. **The file watcher is disabled while benchmarking.** An HMR reload destroys the page's execution
   context and silently kills a multi-minute run; this caused two wasted runs before it was fixed.

---

## Results

24 receipts from the de-leaked `eval` split, 768-token budget, greedy decoding.
Full report: `results/REPORT.md` · dashboard: `npm run dashboard:serve`

| Model | Params | Accuracy | JSON | Exact | p50 | Download | Licence |
|---|---|---|---|---|---|---|---|
| `LFM2.5-VL-450M` | 450M | **74.1%** | 100% | 29.2% | **2.5s** | **304 MB** | LFM1.0 (non-OSI) |
| `SmolVLM-500M-Instruct` | 500M | 63.2% | 100% | 12.5% | 6.4s | 466 MB | Apache-2.0 |
| `SmolVLM2-500M-Video-Instruct` | 500M | 58.5% | 100% | 0% | 6.3s | 466 MB | Apache-2.0 |
| `SmolVLM2-256M-Video-Instruct` | 256M | 24.3% | 75% | 0% | 16.3s | 255 MB | Apache-2.0 |
| `SmolVLM-256M-Instruct` | 256M | 22.4% | 96% | 0% | 5.2s | 255 MB | Apache-2.0 |

Per-field accuracy, which is what should actually drive a model choice:

| Model | merchant | date | total | currency | tax | line_items |
|---|---|---|---|---|---|---|
| `LFM2.5-VL-450M` | 68.8% | 60.8% | **75.0%** | **91.7%** | **54.2%** | **74.2%** |
| `SmolVLM-500M-Instruct` | 65.3% | 83.3% | 41.7% | 62.5% | 29.2% | 34.5% |
| `SmolVLM2-500M-Video-Instruct` | 41.0% | **95.0%** | 18.8% | 79.2% | 12.5% | 22.9% |
| `SmolVLM-256M-Instruct` | 6.3% | 83.3% | 0.0% | 0.0% | 4.2% | 0.0% |
| `SmolVLM2-256M-Video-Instruct` | 30.4% | 66.7% | 0.0% | 0.0% | 0.0% | 1.6% |

Headline conclusions:

- **`LFM2.5-VL-450M` is the best sub-1B option measured** — and it wins on every axis, not just
  accuracy: smaller download, ~2.5× faster, and far better at `total` and `line_items`. At 74% it is
  still a *suggestion* engine; nothing here creates transactions unattended.
- **Architecture mattered more than parameter count.** Changing family moved `total` from 42% to 75%
  while adding parameters inside the SmolVLM family did much less. The 74% ceiling is a property of
  the models tried, not of the sub-1B class.
- **Licence is the catch on the winner.** LFM1.0 is not OSI-approved. The Apache-2.0 fallback is
  `SmolVLM-500M-Instruct` at 63.2%.
- **Both 256M models score 0.0% on `total` and `currency`** — they concatenate the receipt number and
  VAT id into values like `34024218`. That is numeric grounding, which no prompt or grammar can fix.
- **`SmolVLM2` is not simply worse — it trades fields.** It beats SmolVLM-500M on dates (95% vs 83%)
  while losing badly on totals.

---

## Candidates that were tried and excluded

The lineup is four models, but more were investigated. `src/models.js` records each exclusion
with its evidence, and `npm run check:compat` re-verifies that the four benchmarked models still
load, run, and produce coherent output:

```bash
npm run check:compat
```

| Candidate | Params | Size | Licence | Why it is not in the lineup |
|---|---|---|---|---|
| `granite-docling-258M` | 258M | 290 MB | Apache-2.0 | Smallest candidate and loads fine, but it is a document-conversion model, not an instruction follower: asked for receipt JSON it **recites the prompt schema back** instead of reading the image. |
| `FastVLM-0.5B` | 0.5B | 2.1 GB | apple-amlr | Blows the download budget; licence is not OSI-approved. |
| `Qwen2-VL-2B` | 2B | 2.5 GB | Apache-2.0 | Above the sub-1B target. |
| `Qwen2.5-VL-3B` | 3B | — | Apache-2.0 | transformers.js supports the architecture, but **no ONNX weights are published** — not runnable out of the box. |
| `moondream2` | 1.86B | 707 MB | Apache-2.0 | Above the sub-1B target. |

`LFM2.5-VL-450M` was originally in this table as worker-incompatible. **That was a bug in this
harness, not a limitation of the model**, and it has been fixed — see the processor-dispatch note
below. It is now in the lineup and wins.

### A single-field probe makes the core problem vivid

Asked only "what is the total?" on the same receipt (true total `184.89`), the models answered:

| Model | Answer | Correct? |
|---|---|---|
| `LFM2.5-VL-450M` | `"184.89."` | yes |
| `SmolVLM-256M` | `"£184.89"` | yes |
| `SmolVLM-500M` | `"184.89."` | yes |
| `SmolVLM2-256M` | `"29.98"` | no — a line-item price |
| `SmolVLM2-500M` | `"30.81"` | no — the tax amount |

Even in the easiest possible framing, models return confidently wrong numbers. This is numeric
grounding, not format, and it is why `total` must always be user-confirmed.

---

## Findings that will bite you

### A model can be excluded by a harness bug and look like a model limitation

`LFM2.5-VL-450M` was reported as "worker-incompatible" for several rounds. The real cause was a
one-line bug here: `vlm-worker.js` chose the processor's argument order by checking
`processor.constructor.name === 'Lfm2VlProcessor'`, but what `AutoProcessor.from_pretrained` returns
is not always a direct instance of the processor class, so the check never matched and the wrong
order was used — producing an opaque `undefined is not iterable`.

The dispatch now tries `(text, images)` and falls back to `(images, text)` on throw, so it needs no
per-family knowledge at all. The wrong order fails fast, before any expensive work.

**The lesson generalises:** when a model fails to run, verify it is the model's fault before writing
it off. A "known incompatible" entry in a registry is a claim that deserves the same evidence as a
benchmark number. Fixing this changed the recommended model.

### Quantization can silently destroy a model

The `q4f16` and `fp16` ONNX exports of SmolVLM-256M emit degenerate text — `"if if if …"`,
`"-1: -1: -1:"` — on every prompt, while `q4` is correct and fast. This is a defect in those weight
files, not a prompt or template problem: the chat template renders correctly, the image yields the
expected 13 tiles, and the identical prompt succeeds under `q4`.

`uint8`/`int8` are also correct but roughly 15–25× slower per token.

**Always verify a quantization actually produces coherent output before trusting a size comparison.**
`npm run` `scripts/sweep-quantizations.mjs` exists for exactly this.

### The grader, not the model, was the biggest early measurement error

Tiny models emit JSON that is invalid *as a document* while containing individually perfect fields —
`"merchant": "Garcia Supermarket"` sitting next to `"price": +345150-243`. Scoring the document as
unparseable discards the correct merchant and reports 0.

`extractJson` therefore has two recovery stages beyond plain parsing: `truncated-salvage` closes
open brackets when generation hit the token ceiling, and `field-salvage` pulls out individually
well-formed contract keys from otherwise broken output. Re-grading the *same stored output* raised
every model, by as much as **+19 points** on SmolVLM2-500M — enough to change the ranking.

`scripts/rescore.mjs` exists for this: model output is the expensive, durable artifact, so when the
grader changes, re-grade rather than re-run.

### Token budget silently truncates results

Verbose outputs routinely hit `max_new_tokens` mid-`line_items`, which invalidates the entire JSON
document and scores an otherwise-correct extraction 0.0. Run with at least 768 tokens.

### `StructuredOutputProcessor` does not appear to constrain generation in this stack

`@huggingface/transformers-structured-output` exports `StructuredOutputProcessor`, which *extends*
`LogitsProcessorList`, but the generation loop invokes `logits_processor` as a `Callable`. The object
is accepted and then silently ignored: output is byte-identical whether the schema has five
properties or six, and `line_items` appears even when the schema forbids it. **Do not treat the
`--constrain` path as verified grammar enforcement.** It is retained only because it also disables
`repetition_penalty`, which by itself measurably helps the 256M tier.

### Browser constraints

- **Download size, not VRAM, is the product constraint.** A 256M model at `q4` is ~430 MB; the 500M
  is ~800 MB. Neither can block first load on mobile data — fetch lazily, cache, and treat this as
  progressive enhancement over manual entry.
- **`shader-f16` is ~93% available**, absent on older Adreno 5xx/6xx and NVIDIA Maxwell/Pascal. Pick
  the dtype at runtime from `adapter.features`.
- **Safari allows ~1 GB of storage per origin** and evicts script-writable storage after 7 days
  without interaction unless installed as a PWA — users may silently re-download the model.
- **CPU-side image preprocessing is not negligible** (~0.8s against ~4s of decoding), so downscaling
  images before inference is a real optimisation.

---

## Gotchas when extending this

- **Fixtures are gitignored** and regenerated by `npm run fixtures`. The extractor reads parquet over
  HTTP range requests (a few MiB instead of hundreds of MB) and gets the byte-exact original images.
  The datasets-server `/rows` endpoint silently re-encodes `image_clean` to lossy JPEG, so it is not
  equivalent.
- **The `utf8` flag matters** in the parquet reader: left at its default, image `BYTE_ARRAY` columns
  are decoded as strings and the PNG bytes are corrupted into U+FFFD replacement characters with no
  error. Likewise `useOffsetIndex` must be enabled or a row group's entire image column (~92 MiB) is
  transferred to read a handful of rows.
- **`npm run serve` intentionally has no watcher.** Use `npm run serve:watch` when iterating on
  harness code, but never during a benchmark run.
- **Add new candidates to `src/models.js`** and verify with `scripts/check-extra-models.mjs` that the
  repo actually publishes ONNX weights transformers.js can resolve. Most VLMs do not.

---

## License notes

The benchmark harness is original work. Fixtures derive from
`albertobarnabo/synthetic-receipts-ocr` (Apache-2.0). Model weights carry their own licenses —
check before shipping; `SmolVLM-*` is Apache-2.0.
