/**
 * Fetch a small receipt fixture set from the HuggingFace dataset
 * `albertobarnabo/synthetic-receipts-ocr`.
 *
 * Why the datasets-server `/rows` endpoint: it returns the label JSON and a
 * presigned image URL in one response, needs no auth, and needs no Python or
 * parquet decoder. Presigned URLs expire, so we download the bytes in the same
 * pass and never persist the URLs.
 *
 * Output layout (gitignored):
 *   fixtures/<split>/images/<id>.png|jpg
 *   fixtures/<split>/labels/<id>.json     normalised to the benchmark schema
 *   fixtures/<split>/raw/<id>.json        full original `fields` object
 *   fixtures/<split>/manifest.json
 *
 *   node scripts/fetch-fixtures.mjs [--split eval] [--limit 40] [--offset 0]
 */
import { mkdir, writeFile, readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const DATASET = 'albertobarnabo/synthetic-receipts-ocr';
const CONFIG = 'default';
const PAGE = 100; // datasets-server caps `length` at 100.

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const split = arg('split', 'eval');
const limit = Number(arg('limit', '40'));
const offset = Number(arg('offset', '0'));
const variant = arg('variant', 'image_photo'); // image_photo | image_clean
const outRoot = path.join(process.cwd(), 'fixtures', split);

/**
 * Map the dataset's locale-formatted date to ISO.
 * Locales in this dataset use different orders: UK "23/07/2020", DE "07.10.2021".
 */
function toIsoDate(raw, locale) {
  if (!raw || typeof raw !== 'string') return null;
  const s = raw.trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
  const m = s.match(/^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$/);
  if (!m) return null;
  let [, a, b, y] = m;
  if (y.length === 2) y = String(Number(y) >= 70 ? 1900 + Number(y) : 2000 + Number(y));
  const n1 = Number(a);
  const n2 = Number(b);
  // US-style receipts are month-first; UK/DE/IT/FR are day-first.
  const dayFirst = locale !== 'US';
  let day;
  let month;
  if (n1 > 12 && n2 <= 12) { day = n1; month = n2; }
  else if (n2 > 12 && n1 <= 12) { day = n2; month = n1; }
  else if (dayFirst) { day = n1; month = n2; }
  else { month = n1; day = n2; }
  return `${y}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
}

/** Fold the dataset's rich label JSON into the strict benchmark schema. */
function toBenchmarkLabel(fields) {
  const taxes = Array.isArray(fields.taxes) ? fields.taxes : [];
  const taxSum = taxes.length
    ? taxes.reduce((a, t) => a + (Number(t?.amount) || 0), 0)
    : null;
  const lines = Array.isArray(fields.lines) ? fields.lines : [];
  return {
    merchant: fields.merchant ?? null,
    date: toIsoDate(fields.date, fields.locale),
    total: typeof fields.total === 'number' ? fields.total : Number(fields.total) || null,
    currency: fields.currency ?? null,
    tax: taxSum == null ? null : Number(taxSum.toFixed(2)),
    line_items: lines.map((l) => ({
      description: l?.name ?? null,
      qty: l?.qty ?? null,
      unit_price: l?.unit_price ?? null,
      amount: typeof l?.total === 'number' ? l.total : null,
    })),
  };
}

async function fetchRows(pageOffset) {
  const url =
    `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET)}` +
    `&config=${CONFIG}&split=${split}&offset=${pageOffset}&length=${PAGE}`;
  const res = await fetch(url, { headers: { 'user-agent': 'vlm-receipts-bench/0.1' } });
  if (!res.ok) {
    throw new Error(`rows fetch failed ${res.status} ${res.statusText} for offset ${pageOffset}`);
  }
  return res.json();
}

const needed = limit;
const rows = [];
for (let o = offset; rows.length < needed; o += PAGE) {
  const page = await fetchRows(o);
  if (!page.rows?.length) break;
  for (const r of page.rows) {
    rows.push(r.row);
    if (rows.length >= needed) break;
  }
  if (o + PAGE >= (page.num_rows_total ?? 0)) break;
}

console.log(`[fixtures] fetched ${rows.length} label rows from ${DATASET} (${split})`);

await mkdir(path.join(outRoot, 'images'), { recursive: true });
await mkdir(path.join(outRoot, 'labels'), { recursive: true });
await mkdir(path.join(outRoot, 'raw'), { recursive: true });

const manifest = [];
let downloaded = 0;
let failed = 0;

for (const row of rows) {
  const id = row.id;
  const imgField = row[variant] ?? row.image_photo ?? row.image_clean;
  if (!imgField?.src) {
    console.warn(`[fixtures] ${id}: no image url in ${variant}`);
    failed++;
    continue;
  }

  const isJpeg = /\.jpe?g$/i.test(imgField.src.split('?')[0]);
  const ext = isJpeg ? 'jpg' : 'png';
  const imgPath = path.join(outRoot, 'images', `${id}.${ext}`);
  const labelPath = path.join(outRoot, 'labels', `${id}.json`);
  const rawPath = path.join(outRoot, 'raw', `${id}.json`);

  try {
    if (existsSync(imgPath) && existsSync(labelPath)) {
      // Already cached from a previous run; reuse.
      const raw = JSON.parse(await readFile(rawPath, 'utf8'));
      manifest.push(buildEntry(id, imgPath, labelPath, raw, row, ext));
      continue;
    }

    const imgRes = await fetch(imgField.src);
    if (!imgRes.ok) throw new Error(`image ${imgRes.status}`);
    const bytes = new Uint8Array(await imgRes.arrayBuffer());
    await writeFile(imgPath, bytes);

    const fields = JSON.parse(row.fields);
    const label = toBenchmarkLabel(fields);
    await writeFile(labelPath, JSON.stringify(label, null, 2));
    await writeFile(rawPath, JSON.stringify(fields, null, 2));
    manifest.push(buildEntry(id, imgPath, labelPath, fields, row, ext));
    downloaded++;
    process.stdout.write(`\r[fixtures] downloaded ${downloaded} (${id})      `);
  } catch (err) {
    failed++;
    console.warn(`\n[fixtures] ${id} failed: ${err.message}`);
  }
}

function buildEntry(id, imgPath, labelPath, fields, row, ext) {
  return {
    id,
    image: path.relative(process.cwd(), imgPath).replace(/\\/g, '/'),
    label: path.relative(process.cwd(), labelPath).replace(/\\/g, '/'),
    bytes: null,
    ext,
    locale: row.locale ?? fields.locale ?? null,
    degradations: row.degradations ?? null,
    font: row.font ?? null,
    nItems: row.n_items ?? null,
    splitPolicy: row.split_policy ?? null,
  };
}

// Record on-disk sizes so the report can note image weight.
for (const e of manifest) {
  try {
    const { size } = await stat(e.image);
    e.bytes = size;
  } catch { /* ignore */ }
}

await writeFile(
  path.join(outRoot, 'manifest.json'),
  JSON.stringify(
    {
      dataset: DATASET,
      config: CONFIG,
      split,
      variant,
      generatedAt: new Date().toISOString(),
      count: manifest.length,
      entries: manifest,
    },
    null,
    2,
  ),
);

console.log('');
console.log(`[fixtures] wrote ${manifest.length} fixtures to fixtures/${split}/ (downloaded ${downloaded}, failed ${failed})`);
if (manifest.length) {
  const loc = {};
  for (const e of manifest) loc[e.locale ?? '?'] = (loc[e.locale ?? '?'] ?? 0) + 1;
  console.log(`[fixtures] locales: ${JSON.stringify(loc)}`);
  const avg = manifest.reduce((a, e) => a + (e.bytes ?? 0), 0) / manifest.length;
  console.log(`[fixtures] avg image size: ${(avg / 1024).toFixed(0)} KB`);
}
