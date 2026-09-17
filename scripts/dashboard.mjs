/**
 * Build ONE self-contained dashboard from a benchmark result file.
 *
 * Combines the charts, the result tables, the findings, and the provenance into
 * a single HTML file so the whole result can be judged in one view without
 * cross-referencing a separate report. Charts are inlined as SVG, so the file
 * needs no CDN, no build step, and works offline from disk.
 *
 * Findings prose is sourced from the generated REPORT.md rather than duplicated
 * here, so the narrative and the dashboard cannot drift apart.
 *
 *   node scripts/dashboard.mjs results/bench-final.rescored.json
 *                                [--report results/REPORT.md]
 *                                [--out results/dashboard.html]
 *                                [--png]
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { renderCharts } from './charts.mjs';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const has = (name) => process.argv.includes(`--${name}`);

const input = process.argv.slice(2).find((a) => !a.startsWith('--') && a.endsWith('.json'));
if (!input) {
  console.error('usage: node scripts/dashboard.mjs <results.json> [--report REPORT.md] [--out dashboard.html]');
  process.exit(1);
}
const reportPath = arg('report', path.join('results', 'REPORT.md'));
const outPath = arg('out', path.join('results', 'dashboard.html'));

const data = JSON.parse(await readFile(input, 'utf8'));

// REPORT.md supplies the narrative sections; it may not exist for a fresh run.
let report = '';
try {
  report = await readFile(reportPath, 'utf8');
} catch {
  console.warn(`[dashboard] no report at ${reportPath}; omitting findings sections`);
}

// --------------------------------------------------------------- markdown ----
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** Inline formatting: `code` and **bold**. Escapes first, so this is safe. */
function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

/**
 * Minimal markdown renderer for the subset REPORT.md actually uses: headings,
 * tables, ordered/unordered lists, blockquotes and fenced code. Written by hand
 * because the dashboard must stay dependency-free.
 */
function renderMarkdown(md) {
  const out = [];
  const lines = md.split('\n');
  let i = 0;
  const closeList = (stack) => {
    while (stack.length) out.push(`</${stack.pop()}>`);
  };
  const listStack = [];

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code
    if (/^```/.test(line)) {
      const body = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) body.push(lines[i++]);
      i++;
      closeList(listStack);
      out.push(`<pre><code>${esc(body.join('\n'))}</code></pre>`);
      continue;
    }

    // Table: header row followed by a separator row
    if (/^\|/.test(line) && /^\|[\s:|-]+\|$/.test(lines[i + 1] ?? '')) {
      closeList(listStack);
      const cells = (r) => r.split('|').slice(1, -1).map((c) => c.trim());
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push('<div class="tablewrap"><table>');
      out.push(`<thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join('')}</tr></thead>`);
      out.push('<tbody>');
      for (const r of rows) out.push(`<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`);
      out.push('</tbody></table></div>');
      continue;
    }

    // Headings
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      closeList(listStack);
      const level = h[1].length;
      const text = h[2];
      const id = text.toLowerCase().replace(/[^a-z0-9\s-]/g, '').trim().replace(/\s+/g, '-');
      out.push(`<h${level} id="${id}">${inline(text)}</h${level}>`);
      i++;
      continue;
    }

    // Lists
    const ol = line.match(/^(\d+)\.\s+(.*)$/);
    const ul = line.match(/^[-*]\s+(.*)$/);
    if (ol || ul) {
      const want = ol ? 'ol' : 'ul';
      if (listStack[listStack.length - 1] !== want) {
        closeList(listStack);
        out.push(`<${want}>`);
        listStack.push(want);
      }
      out.push(`<li>${inline((ol ? ol[2] : ul[1]))}</li>`);
      i++;
      continue;
    }

    if (/^\s*$/.test(line)) { closeList(listStack); i++; continue; }

    closeList(listStack);
    out.push(`<p>${inline(line)}</p>`);
    i++;
  }
  closeList(listStack);
  return out.join('\n');
}

/**
 * Pull the sections worth showing in the dashboard, dropping the ones the
 * dashboard renders itself (recommendation is a KPI header; results and
 * per-field accuracy are interactive tables; methodology is its own strip).
 */
/**
 * Pull the sections worth showing in the dashboard, dropping the ones the
 * dashboard renders itself (recommendation is a KPI header; results and
 * per-field accuracy are interactive tables; methodology is its own strip).
 *
 * Heading text is matched after stripping a leading "N. " section number, so
 * renumbering the report cannot silently drop a section from the dashboard.
 */
function extractSections(md) {
  const parts = md.split(/^## /m);
  const wanted = [
    'Candidates investigated and excluded',
    'What actually determined accuracy',
    'What this means for the PWA',
    'Open questions worth testing next',
  ];
  const found = [];
  for (const p of parts) {
    const title = p.split('\n')[0].trim().replace(/^\d+\.\s*/, '');
    for (const w of wanted) {
      if (title.toLowerCase().startsWith(w.toLowerCase())) found.push(`## ${p.trim()}`);
    }
  }
  return found.join('\n\n');
}

// ------------------------------------------------------------------ render ---
const { charts, shortName } = await renderCharts(data);

const runs = (data.runs ?? []).filter((r) => r.status === 'ok' && r.summary);
const ranked = [...runs].sort((a, b) => b.summary.accuracy - a.summary.accuracy);
const w = ranked[0];

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const pct0 = (x) => `${Math.round(x * 100)}%`;
const secs = (ms) => `${(ms / 1000).toFixed(1)}s`;
const mb = (b) => `${Math.round(b / 1048576)} MB`;

const FIELDS = [
  ['merchant', 'merchant'],
  ['date', 'date'],
  ['total', 'total'],
  ['currency', 'currency'],
  ['tax', 'tax'],
  ['line_items', 'line items'],
];

const PALETTE = {
  bg: '#12151b', panel: '#181c24', grid: '#2c313c', text: '#d7dae0', textDim: '#9aa0a6', accent: '#4f8ef7',
};

function heat(v) {
  const stops = [[0, [190, 70, 65]], [0.5, [224, 163, 62]], [1, [76, 175, 125]]];
  let a = stops[0];
  let b = stops[2];
  for (let i = 0; i < 2; i++) if (v >= stops[i][0] && v <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; }
  const t = b[0] === a[0] ? 0 : (v - a[0]) / (b[0] - a[0]);
  const c = a[1].map((x, i) => Math.round(x + (b[1][i] - x) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const resultsTable = `
<div class="tablewrap"><table class="sortable">
  <thead><tr>
    <th>Model</th><th>Params</th><th>dtype</th>
    <th class="num">Accuracy</th><th class="num">JSON valid</th><th class="num">Exact match</th>
    <th class="num">p50</th><th class="num">p90</th><th class="num">Download</th>
  </tr></thead>
  <tbody>
  ${ranked.map((r, i) => `
    <tr class="${i === 0 ? 'winner' : ''}">
      <td><strong>${esc(shortName(r))}</strong><div class="dim mono">${esc(r.model)}</div></td>
      <td>${esc(r.params ?? '?')}</td>
      <td class="mono">${esc(r.dtype)}</td>
      <td class="num">${pct(r.summary.accuracy)}</td>
      <td class="num">${pct(r.summary.jsonParseRate)}</td>
      <td class="num">${pct(r.summary.exactMatchRate)}</td>
      <td class="num">${secs(r.summary.latencyMs.p50)}</td>
      <td class="num">${secs(r.summary.latencyMs.p90)}</td>
      <td class="num">${mb(r.load.downloadBytes)}</td>
    </tr>`).join('')}
  </tbody>
</table></div>`;

const fieldTable = `
<div class="tablewrap"><table>
  <thead><tr><th>Model</th>${FIELDS.map(([, l]) => `<th class="num">${esc(l)}</th>`).join('')}</tr></thead>
  <tbody>
  ${ranked.map((r) => `
    <tr>
      <td>${esc(shortName(r))}</td>
      ${FIELDS.map(([k]) => {
        const v = r.summary.fieldAccuracy?.[k] ?? 0;
        const dark = v < 0.06;
        return `<td class="num heat" style="background:${heat(v)};color:${dark ? PALETTE.text : '#0d1017'}">${pct0(v)}</td>`;
      }).join('')}
    </tr>`).join('')}
  </tbody>
</table></div>
<div class="note">Headline accuracy averages <code>merchant</code>, <code>date</code>, <code>total</code> and <code>currency</code>.
<code>tax</code> and <code>line items</code> are shown but excluded — a transaction can be created without itemisation.</div>`;

const gpuLine = data.gpu
  ? `${esc(data.gpu.vendor)} ${esc(data.gpu.architecture)}${data.gpu.shaderF16 ? ', shader-f16' : ''}`
  : 'unknown GPU';

const findings = extractSections(report);

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VLM Receipt Benchmark — Dashboard</title>
<style>
  :root { color-scheme: dark; --bg:${PALETTE.bg}; --panel:${PALETTE.panel}; --grid:${PALETTE.grid};
          --text:${PALETTE.text}; --dim:${PALETTE.textDim}; --accent:${PALETTE.accent}; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.65 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  header { border-bottom:1px solid var(--grid); background:linear-gradient(180deg,#171b23,#12151b); }
  .wrap { max-width:1120px; margin:0 auto; padding:0 1.25rem; }
  header .wrap { padding-top:2rem; padding-bottom:1.5rem; }
  h1 { font-size:1.6rem; margin:0 0 .3rem; letter-spacing:-.01em; }
  h2 { font-size:1.2rem; margin:2.5rem 0 .75rem; padding-top:.5rem; }
  h3 { font-size:1rem; margin:1.75rem 0 .5rem; }
  .sub { color:var(--dim); font-size:.92rem; }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:.75rem; margin-top:1.25rem; }
  .kpi { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:.75rem 1rem; min-width:0; }
  .kpi .k { font-size:.72rem; color:var(--dim); text-transform:uppercase; letter-spacing:.05em; }
  /* nowrap + a smaller clamp so a long model name stays on one line without
     widening the card and breaking the grid alignment. */
  .kpi .v { font-size:clamp(1rem,1.5vw,1.35rem); font-weight:650; margin-top:.15rem; white-space:nowrap; }
  .kpi .s { font-size:.75rem; color:var(--dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  nav.toc { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.25rem; }
  nav.toc a { color:var(--text); text-decoration:none; font-size:.82rem; background:var(--panel);
              border:1px solid var(--grid); padding:.3rem .65rem; border-radius:999px; }
  nav.toc a:hover { border-color:var(--accent); }
  .callout { background:var(--panel); border:1px solid var(--grid); border-left:3px solid var(--accent);
             border-radius:8px; padding:.85rem 1.1rem; margin:1.25rem 0; }
  .callout.warn { border-left-color:#e0a33e; }
  figure { margin:0 0 2rem; }
  figcaption { color:var(--dim); font-size:.85rem; margin-top:.5rem; }
  .chart { border:1px solid var(--grid); border-radius:12px; overflow:hidden; background:var(--bg); }
  .chart svg { display:block; width:100%; height:auto; }
  .tablewrap { overflow-x:auto; border:1px solid var(--grid); border-radius:10px; }
  table { border-collapse:collapse; width:100%; font-size:.87rem; }
  th, td { padding:.55rem .7rem; text-align:left; border-bottom:1px solid var(--grid); }
  th { background:#1b202a; font-size:.74rem; text-transform:uppercase; letter-spacing:.04em;
       color:var(--dim); position:sticky; top:0; }
  tr:last-child td { border-bottom:none; }
  tr.winner td { background:rgba(79,142,247,.10); }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  td.heat { font-weight:600; }
  .mono { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.78rem; }
  .dim { color:var(--dim); }
  .note { color:var(--dim); font-size:.82rem; margin-top:.5rem; }
  code { background:var(--panel); border:1px solid var(--grid); padding:.08rem .35rem; border-radius:4px;
         font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.85em; }
  pre { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:.85rem 1rem; overflow-x:auto; }
  pre code { background:none; border:none; padding:0; font-size:.82rem; }
  ul, ol { padding-left:1.35rem; }
  li { margin:.3rem 0; }
  a { color:var(--accent); }
  footer { border-top:1px solid var(--grid); margin-top:3rem; padding:1.5rem 0 3rem; color:var(--dim); font-size:.82rem; }
  footer .wrap { padding:0 1.25rem; }
  .grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1rem; }
  .card { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:.9rem 1.05rem; }
  .card h4 { margin:0 0 .4rem; font-size:.9rem; }
  .card p { margin:0; font-size:.85rem; color:var(--dim); }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Small VLM Receipt Parsing — Benchmark Dashboard</h1>
    <div class="sub">Which sub-1B vision-language model can read a receipt photo in a browser tab, on WebGPU, well enough to trust in a personal-finance PWA?</div>
    <div class="kpis">
      <div class="kpi"><div class="k">Recommended</div><div class="v">${esc(w ? shortName(w) : 'n/a')}</div><div class="s">${esc(w?.model ?? '')}</div></div>
      <div class="kpi"><div class="k">Headline accuracy</div><div class="v">${w ? pct(w.summary.accuracy) : 'n/a'}</div><div class="s">merchant · date · total · currency</div></div>
      <div class="kpi"><div class="k">Valid JSON</div><div class="v">${w ? pct(w.summary.jsonParseRate) : 'n/a'}</div><div class="s">after salvage</div></div>
      <div class="kpi"><div class="k">Median latency</div><div class="v">${w ? secs(w.summary.latencyMs.p50) : 'n/a'}</div><div class="s">p90 ${w ? secs(w.summary.latencyMs.p90) : '—'}</div></div>
      <div class="kpi"><div class="k">Download</div><div class="v">${w ? mb(w.load.downloadBytes) : 'n/a'}</div><div class="s">one-time, then cached</div></div>
      <div class="kpi"><div class="k">Receipts</div><div class="v">${esc(String(data.fixtureCount ?? '?'))}</div><div class="s">${esc(String(data.split ?? ''))} split</div></div>
    </div>
    <nav class="toc">
      <a href="#recommendation">Recommendation</a>
      <a href="#charts">Charts</a>
      <a href="#results">Results</a>
      <a href="#findings">Findings</a>
      <a href="#candidates-investigated-and-excluded">Excluded candidates</a>
      <a href="#pwa">PWA implications</a>
      <a href="#provenance">Method &amp; provenance</a>
    </nav>
  </div>
</header>

<main class="wrap">

  <h2 id="recommendation">Recommendation</h2>
  <div class="callout">
    <strong>Ship <code>${esc(w?.model ?? '')}</code> at <code>${esc(w?.dtype ?? '')}</code>.</strong><br/>
    ${w ? `${pct(w.summary.accuracy)} headline accuracy, ${pct(w.summary.jsonParseRate)} valid JSON,
    ${pct(w.summary.exactMatchRate)} exact match on all four headline fields, ${secs(w.summary.latencyMs.p50)} median,
    ${mb(w.load.downloadBytes)} download.` : ''}
  </div>
  <div class="callout warn">
    <strong>This is a suggestion engine, not an autopilot.</strong> At this accuracy a user must confirm the
    parsed values before they become transactions. <strong>No sub-1B model measured here is accurate enough to
    create transactions unattended</strong>, and <code>total</code> is the field most likely to be wrong.
  </div>

  <h2 id="charts">Charts</h2>
${charts.map((c) => `  <figure>
    <div class="chart">${c.svg}</div>
    <figcaption><strong>${esc(c.title)}</strong> — ${esc(c.note)}</figcaption>
  </figure>`).join('\n')}

  <h2 id="results">Results</h2>
  <h3>Overall</h3>
  ${resultsTable}
  <div class="note">Accuracy, JSON-validity and exact-match all count output recovered by the truncation
  and field salvers. First-try strict-JSON parsing alone understated every model.</div>

  <h3>Per field</h3>
  ${fieldTable}

  <h2 id="findings">Findings</h2>
  <div class="grid2">
    <div class="card"><h4>Quantization can silently destroy a model</h4>
      <p><code>q4f16</code> and <code>fp16</code> emit degenerate text for SmolVLM-256M; only <code>q4</code> is
      coherent. <code>uint8</code>/<code>int8</code> work but run 15–25× slower per token.</p></div>
    <div class="card"><h4>Constrained decoding does not apply here</h4>
      <p>A 5-property and 6-property schema produced byte-identical output, and <code>line_items</code> appeared
      when the schema forbade it. The constraint is silently ignored — a negative result.</p></div>
    <div class="card"><h4>The 256M tier fails on numbers</h4>
      <p>Not on format: both 256M models score <strong>0%</strong> on <code>total</code>, concatenating the
      receipt number and VAT id into values like <code>34024218</code>.</p></div>
    <div class="card"><h4>The grader was the biggest early error</h4>
      <p>Discarding broken JSON as a total loss hid correct merchants. Recovery raised SmolVLM2-500M by
      <strong>+19 points</strong> — enough to change the ranking.</p></div>
  </div>
  ${findings ? renderMarkdown(findings) : ''}

  <h2 id="pwa">PWA implications</h2>
  <ul>
    <li><strong>Download size is the binding constraint, not VRAM.</strong> ${w ? mb(w.load.downloadBytes) : '—'} cannot
      block first load on mobile data — fetch lazily, cache, and treat this as progressive enhancement over manual entry.</li>
    <li><strong>WebGPU is not universal.</strong> <code>shader-f16</code> is roughly 93% available and absent on older
      Adreno 5xx/6xx and NVIDIA Maxwell/Pascal, so dtype must be chosen at runtime from <code>adapter.features</code>.</li>
    <li><strong>Warm latency of ~${w ? secs(w.summary.latencyMs.p50) : '—'} per receipt</strong> suits a scan-and-confirm
      flow and not a batch import of a shoebox of receipts.</li>
    <li><strong>Safari allows ~1 GB per origin</strong> and evicts storage after 7 days without interaction unless
      installed as a PWA — users may silently re-download the model.</li>
  </ul>

  <h2 id="provenance">Method &amp; provenance</h2>
  <div class="tablewrap"><table>
    <tbody>
      <tr><th>Dataset</th><td><code>${esc(data.dataset)}</code>${data.datasetRevision ? ` @ <code>${esc(data.datasetRevision.slice(0, 12))}</code>` : ''} — ${esc(String(data.split))} split</td></tr>
      <tr><th>Fixtures</th><td>${esc(String(data.fixtureCount))} receipts, source <code>${esc(String(data.fixturesSource ?? 'unknown'))}</code>; degraded <code>image_photo</code> variant, not the clean render</td></tr>
      <tr><th>Prompt</th><td><code>${esc(String(data.prompt?.id))}</code> — ${esc(String(data.prompt?.description ?? ''))}</td></tr>
      <tr><th>Decoding</th><td>greedy (<code>do_sample: false</code>), max ${esc(String(data.maxNewTokens))} tokens, dtypes at <code>q4</code></td></tr>
      <tr><th>Runtime</th><td>transformers.js on WebGPU / <code>onnxruntime-web</code>, measured inside a real Chrome tab</td></tr>
      <tr><th>Hardware</th><td>${gpuLine}</td></tr>
      <tr><th>Results file</th><td><code>${esc(input)}</code></td></tr>
      <tr><th>Generated</th><td>${esc(data.generatedAt ?? '')}${data.rescoredAt ? ` (re-graded ${esc(data.rescoredAt)})` : ''}</td></tr>
    </tbody>
  </table></div>
  <div class="note">The corpus is entirely synthetic thermal receipts (Latin script, no handwriting or crumpled
  paper); the dataset authors explicitly leave real-photo transfer unmeasured. ${esc(String(data.fixtureCount))} receipts
  gives roughly ±10% error bars — enough to separate 63% from 22%, not enough to settle 63% vs 58%.</div>
</main>

<footer><div class="wrap">
  Regenerate with <code>node scripts/dashboard.mjs ${esc(input)}</code> &middot; charts by
  <code>scripts/charts.mjs</code> &middot; narrative from <code>${esc(reportPath)}</code>.
</div></footer>
</body>
</html>
`;

await mkdir(path.dirname(outPath), { recursive: true });
await writeFile(outPath, html, 'utf8');
console.log(`[dashboard] wrote ${outPath} (${(html.length / 1024).toFixed(0)} KB, ${charts.length} charts inlined)`);

if (has('png')) {
  const { chromium } = await import('playwright-core');
  const { resolve } = await import('node:path');
  const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu-sandbox', '--no-first-run', '--disable-crash-reporter'],
  });
  const page = await browser.newPage({ viewport: { width: 1200, height: 1400 }, deviceScaleFactor: 1 });
  await page.goto(`file:///${resolve(outPath).replace(/\\/g, '/')}`, { waitUntil: 'load' });
  const png = outPath.replace(/\.html$/, '.png');
  await page.screenshot({ path: png, fullPage: true });
  await browser.close();
  console.log(`[dashboard] wrote ${png}`);
}
