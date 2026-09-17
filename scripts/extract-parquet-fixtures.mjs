/**
 * Extract receipt fixtures directly from the dataset's parquet shards.
 *
 * Preferred over `fetch-fixtures.mjs` (which uses the datasets-server `/rows`
 * endpoint) for two concrete reasons:
 *
 *  1. FIDELITY. The viewer re-encodes `image_clean` to JPEG server-side, so that
 *     path silently hands us lossy pixels that are NOT what the dataset holds.
 *     Reading the parquet gives the byte-exact original (a lossless grayscale
 *     PNG), which matters because we are measuring sub-1B models where a
 *     compression artifact can flip a digit.
 *  2. COST/RELIABILITY. Only the needed byte ranges are fetched from the CDN
 *     (a few MiB instead of scraping hundreds of presigned, 1-hour-expiring
 *     URLs), and it avoids the undocumented CloudFront 429s on the viewer.
 *
 * Traps handled here (each fails silently if ignored):
 *  - `utf8: false` — otherwise hyparquet decodes the image BYTE_ARRAY as a string
 *    and corrupts the bytes into U+FFFD replacement characters.
 *  - `useOffsetIndex: true` — otherwise a row group's entire image column chunk
 *    (~92 MiB) is transferred to read a handful of rows.
 *  - image columns are nested groups, so the column name is `image_clean`, not
 *    `image_clean.bytes`.
 *  - `n_items` is INT64 -> BigInt, which JSON.stringify refuses to serialize.
 *
 *   node scripts/extract-parquet-fixtures.mjs [--n 24] [--split eval]
 *                                             [--variant photo|clean|both]
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parquetMetadataAsync, parquetReadObjects } from 'hyparquet';
import { compressors } from 'hyparquet-compressors';

const DATASET = 'albertobarnabo/synthetic-receipts-ocr';
/** Pin the revision so fixtures are reproducible even if the dataset moves. */
const REVISION = 'ed46e02b9b1136f6b54847c1d4bce9e94d11e55c';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const split = arg('split', 'eval');
const wanted = Number(arg('n', '24'));
const variantArg = arg('variant', 'photo');
const outRoot = path.join(process.cwd(), 'fixtures', split);

/**
 * Map the dataset's locale-formatted date to ISO.
 * UK "23/07/2020" and DE "07.10.2021" are both day-first; US is month-first.
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
  const taxSum = taxes.length ? taxes.reduce((a, t) => a + (Number(t?.amount) || 0), 0) : null;
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

/** Read width/height straight from PNG IHDR or JPEG SOF markers. */
function imageSize(bytes, ext) {
  const b = bytes;
  if (ext === 'png' && b[0] === 0x89 && b[1] === 0x50) {
    const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
    return { width: view.getUint32(16), height: view.getUint32(20) };
  }
  if (ext === 'jpg' && b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i < b.length - 9) {
      if (b[i] !== 0xff) { i++; continue; }
      const marker = b[i + 1];
      const len = (b[i + 2] << 8) | b[i + 3];
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { height: (b[i + 5] << 8) | b[i + 6], width: (b[i + 7] << 8) | b[i + 8] };
      }
      i += 2 + len;
    }
  }
  return { width: null, height: null };
}

// ------------------------------------------------------------ shard lookup ---
console.log(`[extract] dataset=${DATASET}`);
console.log(`[extract] revision=${REVISION} split=${split} n=${wanted} variant=${variantArg}`);

const treeRes = await fetch(`https://huggingface.co/api/datasets/${DATASET}/tree/${REVISION}/data`);
if (!treeRes.ok) throw new Error(`tree listing failed: HTTP ${treeRes.status}`);
const tree = await treeRes.json();
const shards = tree
  .filter((e) => e.type === 'file' && e.path.startsWith(`data/${split}-`) && e.path.endsWith('.parquet'))
  .sort((a, b) => a.path.localeCompare(b.path))
  .map((e) => ({
    path: e.path,
    size: e.size,
    url: `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/${e.path}`,
  }));

if (!shards.length) throw new Error(`no parquet shards found for split "${split}"`);
console.log(`[extract] ${shards.length} shard(s): ${shards.map((s) => `${s.path} (${(s.size / 1048576).toFixed(1)}MB)`).join(', ')}`);

const COLUMNS = [
  'id', 'image_clean', 'image_photo', 'fields', 'full_text', 'words', 'words_photo',
  'homography', 'locale', 'degradations', 'font', 'n_items', 'split_policy',
];

await mkdir(path.join(outRoot, 'images'), { recursive: true });
await mkdir(path.join(outRoot, 'labels'), { recursive: true });
await mkdir(path.join(outRoot, 'raw'), { recursive: true });

let transferred = 0;
const manifest = [];
let remaining = wanted;

for (const shard of shards) {
  if (remaining <= 0) break;

  // Resolve the redirect once so range requests hit the CDN directly rather
  // than re-paying the hub resolvers bucket on every slice.
  const head = await fetch(shard.url, { method: 'HEAD' });
  const cdnUrl = head.redirected ? head.url : shard.url;
  const byteLength =
    Number(head.headers.get('x-linked-size') ?? head.headers.get('content-length')) || shard.size;

  const cache = new Map();
  const file = {
    byteLength,
    slice(start, end) {
      const key = `${start}-${end}`;
      if (!cache.has(key)) {
        cache.set(
          key,
          (async () => {
            const res = await fetch(cdnUrl, { headers: { Range: `bytes=${start}-${end - 1}` } });
            if (res.status !== 206 && res.status !== 200) {
              throw new Error(`range ${start}-${end} -> HTTP ${res.status}`);
            }
            const buf = Buffer.from(await res.arrayBuffer());
            transferred += buf.byteLength;
            // Copy into a standalone ArrayBuffer; Buffer pooling shares memory.
            return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          })(),
        );
      }
      return cache.get(key);
    },
  };

  const meta = await parquetMetadataAsync(file);
  const take = Math.min(remaining, Number(meta.num_rows));
  console.log(`[extract] ${shard.path}: ${meta.num_rows} rows, reading first ${take}`);

  const rows = await parquetReadObjects({
    file,
    compressors,
    columns: COLUMNS,
    rowStart: 0,
    rowEnd: take,
    useOffsetIndex: true, // ~7MiB instead of ~130MiB for the same rows
    utf8: false, // keep BYTE_ARRAY columns as Uint8Array (image bytes!)
  });

  for (const row of rows) {
    const id = row.id;
    const fields = JSON.parse(row.fields);
    const locale = row.locale;
    const label = toBenchmarkLabel(fields);

    const variants = [];
    if (variantArg === 'both' || variantArg === 'clean') {
      variants.push({ kind: 'clean', bytes: row.image_clean?.bytes, ext: 'png' });
    }
    if (variantArg === 'both' || variantArg === 'photo') {
      variants.push({ kind: 'photo', bytes: row.image_photo?.bytes, ext: 'jpg' });
    }

    const entry = {
      id,
      images: {},
      label: null,
      locale,
      degradations: row.degradations ?? null,
      font: row.font ?? null,
      nItems: row.n_items == null ? null : Number(row.n_items), // INT64 -> BigInt
      splitPolicy: row.split_policy ?? null,
    };

    for (const v of variants) {
      if (!v.bytes || !v.bytes.byteLength) continue;
      const suffix = variantArg === 'both' ? `.${v.kind}` : '';
      const imgRel = `fixtures/${split}/images/${id}${suffix}.${v.ext}`;
      await writeFile(path.join(process.cwd(), imgRel), v.bytes);
      const dims = imageSize(v.bytes, v.ext);
      entry.images[v.kind] = { path: imgRel, bytes: v.bytes.byteLength, ...dims };
    }

    const labelRel = `fixtures/${split}/labels/${id}.json`;
    await writeFile(path.join(process.cwd(), labelRel), JSON.stringify(label, null, 2));
    await writeFile(path.join(process.cwd(), `fixtures/${split}/raw/${id}.json`), JSON.stringify(fields, null, 2));
    entry.label = labelRel;

    // Primary image for the benchmark (whichever variant was requested).
    const primaryKind = variantArg === 'clean' ? 'clean' : 'photo';
    const primary = entry.images[primaryKind] ?? Object.values(entry.images)[0];
    entry.image = primary?.path ?? null;
    entry.ext = primary?.path?.split('.').pop() ?? null;

    manifest.push(entry);
    remaining--;
    if (remaining <= 0) break;
  }
}

await writeFile(
  path.join(outRoot, 'manifest.json'),
  JSON.stringify(
    {
      dataset: DATASET,
      revision: REVISION,
      config: 'default',
      split,
      variant: variantArg,
      source: 'parquet-range-requests',
      generatedAt: new Date().toISOString(),
      count: manifest.length,
      entries: manifest,
    },
    null,
    2,
  ),
);

const avg = manifest.reduce((a, e) => a + (e.image ? e.images[Object.keys(e.images)[0]].bytes : 0), 0) / (manifest.length || 1);
console.log(`\n[extract] wrote ${manifest.length} fixtures to fixtures/${split}/`);
console.log(`[extract] transferred ${(transferred / 1048576).toFixed(2)} MiB from the CDN`);
console.log(`[extract] avg primary image: ${(avg / 1024).toFixed(0)} KB`);
const loc = {};
for (const e of manifest) loc[e.locale ?? '?'] = (loc[e.locale ?? '?'] ?? 0) + 1;
console.log(`[extract] locales: ${JSON.stringify(loc)}`);
if (manifest[0]?.images) {
  const k = Object.keys(manifest[0].images)[0];
  console.log(`[extract] sample ${manifest[0].id} ${k}: ${JSON.stringify(manifest[0].images[k])}`);
}
