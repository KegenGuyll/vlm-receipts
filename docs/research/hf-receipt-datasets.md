# HuggingFace datasets for benchmarking small VLMs on receipt parsing

Target schema: `{merchant, date, total, currency, tax, line_items:[{description, qty, unit_price, amount}]}`

All facts below were verified by calling the HF datasets-server REST API and `huggingface.co/api/datasets/*`
**anonymously** (no token). Nothing here is from memory.

---

## 0. Does the datasets-server REST API work anonymously? — YES

Verified with zero auth headers, HTTP 200 on all of these:

| Endpoint | Result |
|---|---|
| `/splits?dataset=…`, `/size?dataset=…`, `/parquet?dataset=…` | 200 anonymous |
| `/first-rows?dataset=…&config=…&split=…` | 200 anonymous |
| `/rows?dataset=…&config=…&split=…&offset=N&length=100` | 200 anonymous, `num_rows_per_page: 100` |
| `/filter?…&where=…` | 200 path exists but **timed out (>30 s)** in testing — do not depend on it |
| `huggingface.co/api/datasets/{id}` and `/tree/main/<dir>` | 200 anonymous (`gated:false`, `private:false`; LFS sizes exposed) |

A Node.js downloader with **no Python** is fully viable via `/rows` (which returns image **presigned URLs** inline)
or `/parquet` (direct LFS-backed `.parquet` URLs).

Known non-anonymous / broken cases hit during verification:
- `/rows?dataset=jsdnrs/sroie` → **401** — repo does not exist.
- `/splits?dataset=hperror/wild_receipt` → **401** — repo does not exist.
- `/splits?dataset=mychen76/receipt_ocr` → **401** — repo does not exist.
- `/size?dataset=tunachiu/sroie` → **501** "runs arbitrary python code" — script-based, viewer unsupported.
- `/size?dataset=ryanznie/SROIE_2019_with_labels` → **500 `UnicodeDecodeError`**; parquet conversion failed,
  0 bytes. **Unusable** via datasets-server.
- `/size?dataset=moeinrahimi/receipts` → **500 "The dataset is empty."**

---

## 1. Comparison table

Image-hosting column = whether pixels actually live on the Hub in a loadable parquet/file, verified via
`/parquet` byte sizes and `/rows` asset URLs (not just the repo's stated file list).

| Repo id | Task | Rows (splits) | Images on Hub? | Annotation format | Field mapping to schema | License | Disk | `load_dataset` | Config / splits |
|---|---|---|---|---|---|---|---|---|---|
| **`albertobarnabo/synthetic-receipts-ocr`** | KIE + OCR + bbox (synthetic) | **32,000** (30,000 train / 2,000 eval) | **YES** — 19 `data/*.parquet`, 4.08 GB | `fields` = **JSON string** with full KIE; `full_text`; `words`/`words_photo` word boxes; `homography` | **ALL 6 fields natively present** | `apache-2.0` | 4.08 GB parquet (4.30 GB in RAM) | yes | `default` / `train`, `eval` |
| **`naver-clova-ix/cord-v2`** | KIE / OCR | 1,000 (800 train / 100 val / 100 test) | **YES** — 6 `data/*.parquet`, 2.31 GB | `ground_truth` = JSON string: `gt_parse`, `meta`, `valid_line[].{words[{quad,text}],category}`, `roi` | total ✅, tax ✅, line_items (desc+qty+amount) ✅, **merchant ❌, date ❌, currency ❌** | `cc-by-4.0` | 2.31 GB parquet | yes | `default` / `train`, `validation`, `test` |
| **`rth/sroie-2019-v2`** | KIE + OCR boxes (real) | 973 (626 train / 347 test) | **YES** — 2 `data/*.parquet`, 501.6 MB | `objects.bbox` + `objects.text` + `objects.entities{company,date,address,total}` | merchant ✅, date ✅, total ✅; **currency ❌, tax ❌, line_items ❌** (recoverable by parsing `text`) | `cc-by-2.0` (per card) | 501.6 MB | yes | `default` / `train`, `test` |
| `arvindrajan92/sroie_document_understanding` | Token-classification + boxes (real, SROIE-derived) | 652 (**train only**) | YES — 1 parquet, 217 MB | `ocr[] = {box, label, text}`; labels: `company`, `address`, `date`, `total`, **`line_description`**, **`line_total`**, `other` | merchant ✅, date ✅, total ✅, line_items (desc+amount) ✅; **currency ❌, tax ❌, qty/unit_price ❌** | `mit` | 217 MB | yes | `default` / `train` only |
| `mychen76/invoices-and-receipts_ocr_v2` | OCR + KIE (real photos) | 3,238 (2,843 train / 225 test / 170 valid) | YES — 4 parquet, 430 MB | `parsed_data` (JSON string, double-escaped) + `raw_data` (`ocr_words`, `ocr_boxes` w/ confidence) | total ✅, tax ✅ (when printed), line_items (name+qty+amount, sometimes unit price) ✅; **merchant ❌, date ❌, currency ❌** | **none stated** | 430 MB | yes | `default` / `train`, `test`, `valid` |
| `kaydee/wildreceipt` | 25-class token classification (real) | 1,739 (1,267 train / 472 test) | YES — 3 parquet, 1.37 GB | `words[]`, `bboxes[]`, `ner_tags[]` over 25 classes | all keys exist (`Store_name_*`, `Date_*`, `Prod_item_*`, `Prod_price_*`, `Subtotal_*`, `Tax_*`, `Total_*`) but values are **noisy OCR text**, key↔value pairing needs inference; **no currency** | **none stated** | 1.37 GB | yes | `default` / `train`, `test` |
| `Theivaprakasham/wildreceipt` | Same data, **no images on Hub** | 1,739 (1,267 / 472) | **NO** — parquet is 1.16 MB text-only; `image_path` is an HF worker cache path; 185 MB raw repo | same as above | same | `apache-2.0` | 185 MB raw | yes | `WildReceipt` / `train`, `test` |
| `darentang/sroie` | NER only | 973 (626 / 347) | **NO** — repo contains **only `sroie.py`** (usedStorage 31 MB); script pulls a zip from **Google Drive** | `words` + normalized bboxes + `ner_tags` (`O/B/I-COMPANY/DATE/ADDRESS/TOTAL`) | merchant ✅, date ✅, total ✅; **no currency/tax/line_items** | none stated | parquet 1.0 MB (text) + external drive zip | script only, needs `gdown` | `sroie` / `train`, `test` |
| `cdek-ocr/receipt-ocr-ru` | KIE + bbox (real, Russian) | 999 (699/149/151) | YES | `seller_text/_bbox`, `inn_text/_bbox`, `date_text/_bbox`, `total_text/_bbox`, `item_texts[]`, `item_quantities[]`, `item_prices[]`, `item_sums[]` | merchant ✅, date ✅, total ✅, **line_items w/ qty + unit_price + amount ✅**; tax ❌, currency ❌ | `mit` | 1.68 GB | card warns it "is not intended to be loaded directly via `load_dataset()`" though the viewer works | `default` / `train`, `val`, `test` |
| `asafd60/he-synth-receipt-noisy` | OCR + KIE (synthetic, Hebrew) | 5,000 (4,750 / 250) | YES | `text` = Python-repr dict: `products[{code,name,quantity,kg_price,price_payed}]`, `meta_data{date,entity,address,total_without_maam,final_total,…}` | merchant ✅, date ✅, total ✅, line_items ✅, tax = `final_total − total_without_maam` ✅; **currency ❌** | none stated | 232 MB | yes | `default` / `train`, `test` |
| `toppnoche/receipts-finetune-v3` | KIE (synthetic, Indian) | 20,600 (16,480/2,060/2,060) | YES | `parsed_data` struct: `restaurant_name`, `restaurant_address`, `total_paid_amount`, `date_of_bill`, `time_of_bill`, `gst_number` | merchant ✅, date ✅, total ✅; **line_items ❌, tax ❌ (only GST number), currency ❌** | none stated | 7.94 GB | yes | `default` / `train`, `validation`, `test` |
| `wasanx/receipt-4k` | imagefolder (synthetic Thai/Eng) | ~4,000 | imagefolder, no parquet → viewer limited | none documented | ❌ | none stated | — | imagefolder | `default` / `train`, `validation` |
| `sav7669/sroie_data_set` | imagefolder | **100** | yes | **1 column, no labels** | ❌ | `openrail` | 34 MB | yes | `default` / `train` |
| `UniqueData/ocr-receipts-text-detection` | Detection only | **20** | yes | 5-class shapes | ❌ | **`cc-by-nc-nd-4.0`** (non-commercial, no derivatives) | 55 MB | yes | `train` |
| `SZLHOLDINGS/governed-receipts-bench` | — | **7** | no | 7 columns | ❌ | none | 5 KB | yes | `train` |
| `ryanznie/SROIE_2019_with_labels` | — | viewer **broken** | — | — | ❌ | none | parquet 0 bytes | viewer fails | — |

**Does not exist** (401 anonymous): `jsdnrs/sroie`, `hperror/wild_receipt`, `mychen76/receipt_ocr`.

Sources: [albertobarnabo/synthetic-receipts-ocr](https://huggingface.co/datasets/albertobarnabo/synthetic-receipts-ocr) ·
[cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) · [rth/sroie-2019-v2](https://huggingface.co/datasets/rth/sroie-2019-v2) ·
[arvindrajan92/sroie_document_understanding](https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding) ·
[mychen76/invoices-and-receipts_ocr_v2](https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v2) ·
[kaydee/wildreceipt](https://huggingface.co/datasets/kaydee/wildreceipt) · [Theivaprakasham/wildreceipt](https://huggingface.co/datasets/Theivaprakasham/wildreceipt) ·
[darentang/sroie](https://huggingface.co/datasets/darentang/sroie) · [cdek-ocr/receipt-ocr-ru](https://huggingface.co/datasets/cdek-ocr/receipt-ocr-ru) ·
[asafd60/he-synth-receipt-noisy](https://huggingface.co/datasets/asafd60/he-synth-receipt-noisy) · [toppnoche/receipts-finetune-v3](https://huggingface.co/datasets/toppnoche/receipts-finetune-v3)

---

## 2. Top candidate #1 — `albertobarnabo/synthetic-receipts-ocr` (schema-exact)

**The only dataset found that contains all six target fields natively and exactly.**

- Config `default`; splits `train` (30,000) and `eval` (2,000). Total **32,000 receipts**, each shipped
  **twice** (`image_clean` PNG + `image_photo` JPEG) → 64,000 images, 4.08 GB download.
- Verified anonymously: `/splits` (train, eval), `/size` (32,000 rows, 13 columns, 4,082,419,842 B),
  `/parquet` (19 files: 17 train + 2 eval), `/rows` (both `image_clean` and `image_photo` returned as live
  presigned `src` URLs), `/tree/main/data` (LFS oids + byte sizes).
- License: **`apache-2.0`**. Language tags `en, de, it, fr`; README states 5 locales (US/UK/DE/IT/FR).

### Real `fields` JSON (row `eval-000000`, verbatim, abbreviated)
```json
{"locale":"UK","merchant":"GARCIA SUPERMARKET","address":"37 HIGH STREET","phone":null,
 "tax_id":"VAT NO GB878076490","date":"23/07/2020","time":"13:11","receipt_no":"3-43024-218",
 "cashier":"S. PATEL","currency":"GBP",
 "lines":[{"name":"KB BOHO GREY FUZZY FAUX","qty":2,"unit_price":39.98,"total":79.96,
           "tax_rate":0.2,"source_title":"KB & Me Boho Grey Fuzzy Faux Fur…"}],
 "subtotal":null,"taxes":[{"label":"VAT 20%","rate":0.2,"amount":30.81}],"tax_included":true,
 "total":184.89,"payment":"MASTERCARD","tendered":null,"change":null,"loyalty":"LOYALTY PTS +85"}
```
A DE row (`eval-001179`) returns `"currency":"EUR"`, `"total":81.42`, `"date":"07.10.2021"`,
`"taxes":[{"label":"MwSt 19%","rate":0.19,"amount":13.0}]`, `"tax_included":true`.

### Exact mapping → strict schema
```js
const f = JSON.parse(row.fields);
{
  merchant:  f.merchant,                          // direct
  date:      toISO(f.date, f.locale),             // "23/07/2020" / "07.10.2021" -> ISO, needs locale-aware parse
  total:     f.total,                             // number, already exact
  currency:  f.currency,                          // ISO 4217 string: "GBP" | "EUR" | "USD" | ...
  tax:       f.taxes.reduce((s,t)=>s+t.amount,0), // sum of per-class taxes; label+rate preserved
  line_items: f.lines.map(l=>({
    description: l.name, qty: l.qty, unit_price: l.unit_price, amount: l.total }))
}
```
Extra available and worth keeping: `tax_included` (critical — VAT-inclusive vs US sales-tax-exclusive differ),
`source_title` (unabbreviated product name), `receipt_no`, `payment`, `tendered`, `change`, `address`,
`phone`, `tax_id`, `cashier`, `loyalty`, `n_items`, `locale`, `degradations`, `font`, `split_policy`.

### Usable images
All **32,000**. The `eval` split is deliberately de-leaked: `split_policy: "heldout-font+vocab+merchants"` —
eval shares no font, no product title, and no merchant surname with train (README). Use `eval` (2,000) as the
benchmark split and `train` for few-shot/dev.

### Caveats (stated by the authors, and real)
- **100% synthetic.** Monospace thermal-style only; no proportional fonts, no handwriting, no crumple,
  no logos/barcodes; Latin script; 5 locales.
- README explicitly says transfer to real receipt photos is **"plausible but unmeasured."**
- `tax_id` is `null` on US receipts; `subtotal` is `null` on some rows — respect the documented contract
  ("a field is non-null iff its value is printed on the image").
- **No drop-in HF parquet dataset card is *not* an issue; but `full_text` is only 4 locales of formatting** —
  if your app is US/EU-only this is fine; for other regions you need another source.
- Generator ships in-repo (`generator/*.py`, OFL fonts) — but it is **Python**; irrelevant for a Node downloader
  since the rendered parquet already exists.

---

## 3. Top candidate #2 — `naver-clova-ix/cord-v2` (real photos, strong tax + line items)

- `license: cc-by-4.0`, `gated:false`. Config `default`; splits `train` 800 / `validation` 100 / `test` 100.
- Anonymous `/parquet`: 6 files, **2,307,284,272 B total**; images embedded. `/rows` returns working
  `assets/.../image/image.jpg` presigned URLs → images are on the Hub.
- `ground_truth` is a JSON **string**. Verified structure from `test` rows 0–4, 14–17:
  `gt_parse.menu` (object *or* array: `{nm, cnt, unitprice, price, itemsubtotal, sub}`),
  `gt_parse.sub_total{subtotal_price, discount_price, service_price, tax_price, etc}`,
  `gt_parse.total{total_price, cashprice, changeprice, creditcardprice, menuqty_cnt, total_etc}`,
  plus `meta{version,split,image_id,image_size}`, `valid_line[{words[{quad{...},text}], category, group_id}]`, `roi`.

### Mapping → strict schema
```js
const g = JSON.parse(row.ground_truth).gt_parse;
const menus = Array.isArray(g.menu) ? g.menu : [g.menu];
{
  merchant:  null,                         // NOT PRESENT — see strategy below
  date:      null,                         // NOT PRESENT — see strategy below
  total:     num(g.total.total_price),
  currency:  null,                         // NOT PRESENT (IDR in practice, but never annotated)
  tax:       num(g.sub_total?.tax_price),  // often present, sometimes absent -> null
  line_items: menus.filter(Boolean).map(m=>({
    description: m.nm,
    qty:         num(m.cnt),               // often absent -> default 1
    unit_price:  m.unitprice ? num(m.unitprice) : (num(m.price)/ (num(m.cnt)||1)),
    amount:      num(m.price)
  }))
}
```
Gotchas seen in real rows: numbers are locale-formatted (`"60.000"`, `"28,000"`, `"24,000"`) — **the separator
is inconsistent across rows**, so you need a heuristic (dot vs comma, and small values like `"5.455"` = 5455).
`cnt` is sometimes `"2"`, sometimes `"1X"`, sometimes absent. Sub-items live in `menu[].sub`.
`menu` is an object for single-item receipts and an array otherwise.

**Explicit gaps: no merchant, no date, no currency.** Strategy: use CORD only for the
`line_items + tax + total` axis of the benchmark; do **not** score merchant/date/currency on it.
For date you *can* mine `meta.image_id`-adjacent text in `valid_line` (no `date` category exists — the 25
categories seen are `menu.*`, `sub_total.*`, `total.*` only), so honestly: **date is unrecoverable from CORD**.
For currency, hardcode `IDR` only if you accept an assumption — say so in your eval card.

---

## 4. Top candidate #3 — `rth/sroie-2019-v2` + `arvindrajan92/sroie_document_understanding` (real, merchant/date/total exact)

Use these **together**: `rth/sroie-2019-v2` gives exact `entities` for merchant/date/total on 973 real
scanned receipts; `arvindrajan92` adds `line_description` / `line_total` labels on 652 of the same corpus.

**`rth/sroie-2019-v2`** — `license: cc-by-2.0` per card. Config `default`; `train` 626 / `test` 347 = **973**.
Anonymous `/parquet`: 2 files (318,815,373 + 182,810,878 = **501,626,251 B**), images embedded.
Verified features: `image`, `objects.bbox` (int64 sequences), `objects.text` (string seq),
`objects.entities{company, date, address, total}` — all flat strings, no JSON parsing needed.

```js
{
  merchant: row.objects.entities.company,   // e.g. "OJC MARKETING SDN BHD"
  date:     toISOflexible(row.objects.entities.date), // "15/01/2019", "09/02/2018", "07 MAR 2018", "03 MAR 19"
  total:    num(row.objects.entities.total),// mostly clean ("193.00"); one row observed "$8.20" with a $ sign
  currency: "MYR",                          // NOT ANNOTATED. SROIE is Malaysian; "RM" appears in text, "$" on a few rows
  tax:      null,                           // NOT ANNOTATED — parse from objects.text via /GST|TAX|VAT/
  line_items: null                          // NOT ANNOTATED — parse from objects.text columns
}
```
`bbox` ordering is `[[x1,y1,x2,y2],[...]]` with `y` ascending, so rows are readable top-to-bottom.

**`arvindrajan92/sroie_document_understanding`** — `license: mit`. Config `default`; **`train` only, 652 rows**,
217,146,103 B. Features `image`, `ocr[] = {box (4×2 float), label (string), text}`. Verified labels in
practice: `company`, `address`, `date`, `total`, `line_description`, `line_total`, `other`. This is the only
**real-photo** SROIE mirror found that labels line items. No qty/unit-price label, no tax label, no currency,
and 321 fewer receipts than SROIE-v2 — merge on receipt identity or just benchmark both separately.

**Explicit gaps for the SROIE pair: no currency, no tax field, no structured line items.**
Strategy: (a) hardcode `MYR` and validate against a `text`-regex for `RM`; (b) regex-extract tax from
`objects.text` lines matching `TOTAL GST`, `GST @6%`, `TAX`, `TOTAL INCL .6% GST`, `TAX\(RM\)` — several rows in
`test` show the pattern explicitly (e.g. row 2: `"TOTAL GST(RM) :", "24.69"`; row 21: `"GST @6%: $0.46"`);
(c) build `line_items` only where the receipt prints `DESCRIPTION QTY PRICE AMOUNT` headers — for the
`arvindrajan92` subset use the `line_description`/`line_total` labels and set `qty`/`unit_price` to `null`.
Expect a materially lower ceiling on the line-item axis for SROIE than for CORD.

---

## 5. Concrete Node.js download plan (no Python)

### A. Ground truth + images via `/rows` (works for all top-3)

```js
const BASE = 'https://datasets-server.huggingface.co';

async function* paginate(dataset, config, split) {
  let offset = 0, total = Infinity;
  while (offset < total) {
    const u = `${BASE}/rows?dataset=${encodeURIComponent(dataset)}` +
              `&config=${config}&split=${split}&offset=${offset}&length=100`;
    const r = await fetch(u);                       // no Authorization header needed
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    const j = await r.json();
    total = j.num_rows_total;
    for (const { row_idx, row } of j.rows) yield { row_idx, row };
    offset += j.rows.length;
    if (!j.rows.length) break;
  }
}

async function saveImage(cell, outPath) {           // cell = {src,height,width}
  const r = await fetch(cell.src);                  // presigned S3 URL — download immediately
  if (!r.ok) throw new Error(`${r.status} for ${outPath}`);
  await fs.writeFile(outPath, Buffer.from(await r.arrayBuffer()));
}

// albertobarnabo/synthetic-receipts-ocr
for await (const { row } of paginate('albertobarnabo/synthetic-receipts-ocr', 'default', 'eval')) {
  await saveImage(row.image_photo, `img/${row.id}_photo.jpg`);
  await saveImage(row.image_clean, `img/${row.id}_clean.png`);
  const gt = JSON.parse(row.fields);               // strict-schema ground truth
  await fs.writeFile(`gt/${row.id}.json`, JSON.stringify(gt));
}
```

Useful counts for capacity planning (100 rows/request):
`albertobarnabo` = 320 requests · `cord-v2` = 10 · `rth/sroie-2019-v2` = 10 ·
`mychen76/…_v2` = 33 · `kaydee/wildreceipt` = 18 · `toppnoche/…-v3` = 206.

**Two hard caveats on `/rows`:**
1. **Presigned URLs expire** (`?Expires=…&Signature=…&Key-Pair-Id=…`). Fetch them in the same pass; do not
   persist the URLs. Re-running `/rows` mints fresh ones.
2. **Do not use `/filter`** — it timed out at 30 s in testing.

### B. Faster bulk path
`GET /parquet?dataset=…` (verified 200 anonymous) returns direct LFS-backed URLs such as
`https://huggingface.co/datasets/albertobarnabo/synthetic-receipts-ocr/resolve/refs%2Fconvert%2Fparquet/default/eval/0000.parquet`.
Read them in Node with `hyparquet` or `parquet-wasm` (pure JS/WASM, no Python). Better than `/rows` for the
32k dataset (19 files vs 320 paged requests). The `fields` column is a plain UTF-8 JSON string, so no exotic
decoding is needed.

---

## 6. Final recommendation

| Rank | Dataset | Use it for | Because | Not for |
|---|---|---|---|---|
| **1** | `albertobarnabo/synthetic-receipts-ocr` (eval = 2,000) | **All six fields, exact JSON, strict-schema scoring** | Only source with native `merchant` + `date` + `total` + `currency` + `taxes[]` + `lines[{name,qty,unit_price,total,tax_rate}]`; 32k rows; apache-2.0; de-leaked eval split | Real-photo realism — synthetic only, and transfer is unmeasured |
| **2** | `naver-clova-ix/cord-v2` (test = 100) | **Real photos: line_items + tax + total** | Real photographed Thai receipts; exactly reproduces the line-item arithmetic; boxes for grounding | merchant / date / currency — **absent**; inconsistent decimal separators |
| **3** | `rth/sroie-2019-v2` (test = 347) + `arvindrajan92/sroie_document_understanding` (652, train) | **Real photos: merchant + date + total** (SROIE also gives box-level OCR text) | Real scanned English receipts; exact `company`/`date`/`total`; 973 images | currency / tax / line_items as *fields*; the `darentang/sroie` mirror (Google-Drive-only images, text-only parquet) should be avoided |

**Recommended benchmark design.** Score the strict schema on `albertobarnabo/synthetic-receipts-ocr`
`eval` (2,000) as the primary, all-six-fields leaderboard — it is the only place a strict JSON schema can be
scored end-to-end without assumptions. Report it as a **synthetic** number. Then run a second, real-photo panel
with **field-masked** metrics: CORD `test` (100) scored on `line_items`/`tax`/`total` only, and SROIE-v2 `test`
(347) scored on `merchant`/`date`/`total` only. Any single aggregate number over all six fields across CORD or
SROIE would be reporting fabricated merchant/currency values.

**If you need a real-photo, tax-bearing, line-item bearing set at more than 1k scale, add
`mychen76/invoices-and-receipts_ocr_v2`** (3,238 real receipts, 430 MB, `parsed_data.line_items[]` +
`subtotal.tax` + `total.total`) — but note it has **no license tag** and **no merchant/date/currency**, and its
`parsed_data` values are double-escaped JSON strings (verify around 20% of rows by hand before trusting it).

**Do not use:** `kaydee/wildreceipt` / `Theivaprakasham/wildreceipt` (target values are noisy OCR text and
the key↔value pairing is not annotated — no license on `kaydee`'s copy either), `darentang/sroie`
(external Google Drive, text-only parquet), `wasanx/receipt-4k` and `sav7669/sroie_data_set` (no labels),
`UniqueData/ocr-receipts-text-detection` (20 images, `cc-by-nc-nd-4.0`), `SZLHOLDINGS/governed-receipts-bench`
(7 rows), `ryanznie/SROIE_2019_with_labels` (viewer conversion fails), `toppnoche/receipts-finetune-v3`
(no line items, no currency, no tax; 7.9 GB for 6 usable fields).

---

## 7. Confidence & gaps

**High confidence (directly observed, anonymous HTTP 200):**
- Row counts, split names, column names/dtypes, parquet file sizes and URLs for every repo in the table.
- The complete `fields` JSON schema of `albertobarnabo/synthetic-receipts-ocr`, from two different locales
  (UK and DE), including `currency` and `taxes[]`.
- CORD `gt_parse` structure from 8 distinct test rows — merchant/date/currency are genuinely absent.
- SROIE-v2 `entities` labels from 11 test rows; `darentang/sroie`'s only sibling file is `sroie.py` and its
  `_URLS` points at `drive.google.com` (read from the raw script).
- That `/splits`, `/size`, `/parquet`, `/rows`, `/first-rows` all work **without authentication**.

**Medium confidence / not fully verified:**
- **Exact image-hosted-ness for `darentang/sroie`**: its `num_bytes_original_files` (455,664,162) exceeds its
  repo `usedStorage` (31,366,085), which strongly implies the pixels came from the script's external download,
  but I did not fetch the Drive zip to confirm it is currently reachable.
- **Locale distribution** of `albertobarnabo/synthetic-receipts-ocr` — I confirmed UK and DE rows and the
  README's claim of US/UK/DE/IT/FR, but did not enumerate all 32,000 rows. Do not assume an even 6,400/locale.
- **Licenses**: `mychen76/invoices-and-receipts_ocr_v2`, `kaydee/wildreceipt`, `darentang/sroie`,
  `toppnoche/receipts-finetune-v3`, `asafd60/he-synth-receipt-noisy`, `wasanx/receipt-4k` carry **no license
  tag**. Absence is not permission — treat as unusable for anything shipped.
- `rth/sroie-2019-v2`'s card asserts `cc-by-2.0`; the original ICDAR SROIE terms come from
  [rrc.cvc.uab.es](https://rrc.cvc.uab.es/?ch=13) and I did not re-read them. Confirm before commercial use.
- I did **not** empirically download a full parquet file end-to-end (would have been hundreds of MB), so
  "publicly downloadable without login" rests on `gated:false` + anonymous datasets-server 200s + public
  `/tree` listings, not on a completed byte-range transfer.

**Out of scope / not checked:** actual VLM benchmark numbers on any of these; image quality/aesthetics of the
synthetic renders beyond the two samples inspected; whether `albertobarnabo`'s generator reproduces the
published split deterministically (the README claims `(seed, index)` purity, unverified).
