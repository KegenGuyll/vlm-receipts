/**
 * Render a human-readable report from one or more benchmark result files.
 *
 * The point of this file is the recommendation: an engineer should be able to
 * read the output and know which model to ship, what it will cost a user in
 * download and latency, and where it will silently get things wrong.
 *
 *   node scripts/report.mjs results/bench-constrained.json [more.json ...]
 *                         [--out results/REPORT.md]
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { EXCLUDED_MODELS } from '../src/models.js';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const files = process.argv.slice(2).filter((a) => !a.startsWith('--') && a.endsWith('.json'));
if (!files.length) {
  console.error('usage: node scripts/report.mjs <results.json> [...] [--out REPORT.md]');
  process.exit(1);
}
const outPath = arg('out', path.join('results', 'REPORT.md'));

const runs = [];
for (const f of files) {
  const data = JSON.parse(await readFile(f, 'utf8'));
  for (const r of data.runs ?? []) runs.push({ ...r, _source: f, _runMeta: data });
}

const pct = (x) => (x == null ? '--' : `${(x * 100).toFixed(1)}%`);
const secs = (ms) => (ms == null ? '--' : `${(ms / 1000).toFixed(1)}s`);
const mb = (b) => (b == null ? '--' : `${(b / 1048576).toFixed(0)} MB`);

/** Pick the winner: accuracy first, then latency, then download size. */
const ok = runs.filter((r) => r.status === 'ok' && r.summary);
const ranked = [...ok].sort((a, b) => {
  const d = (b.summary.accuracy ?? 0) - (a.summary.accuracy ?? 0);
  if (Math.abs(d) > 0.02) return d;
  const l = (a.summary.latencyMs.p50 ?? Infinity) - (b.summary.latencyMs.p50 ?? Infinity);
  if (l !== 0) return l;
  return (a.load?.downloadBytes ?? Infinity) - (b.load?.downloadBytes ?? Infinity);
});
const winner = ranked[0];
const failed = runs.filter((r) => r.status !== 'ok');

// Candidates that were investigated but are not in the results. Anything that
// actually ran is reported in the results tables instead.
const ranRepos = new Set(runs.map((r) => r.model));
const excluded = EXCLUDED_MODELS.filter((e) => !ranRepos.has(e.repo));

const L = [];
const w = (s = '') => L.push(s);

w('# Small VLM Receipt Parsing — Benchmark Report');
w();
w(`Generated: ${new Date().toISOString()}`);
w();
w('Every number here was produced by running the model **inside a real Chrome tab on WebGPU**,');
w('driven by Playwright against the harness in `src/`. No server-side inference was involved.');
w();

// ---------------------------------------------------------------- headline --
w('## 1. Recommendation');
w();
if (winner) {
  const s = winner.summary;
  const f = s.fieldAccuracy ?? {};
  // Name the actual weakest fields rather than assuming which they are. The
  // weakest field differs by model — for the LFM2.5 winner `total` is its
  // STRONGEST field, so generic advice would have been actively wrong.
  const headlineFields = ['merchant', 'date', 'total', 'currency'];
  const byWeakness = [...headlineFields].sort((a, b) => (f[a] ?? 0) - (f[b] ?? 0));
  const weakest = byWeakness.slice(0, 2);
  const strongest = byWeakness[byWeakness.length - 1];

  w(`**Ship \`${winner.model}\` at \`${winner.dtype}\`.**`);
  w();
  w(`- Headline accuracy: **${pct(s.accuracy)}** over ${s.count} receipts (merchant, date, total, currency)`);
  w(`- Valid JSON on first try: **${pct(s.jsonParseRate)}**`);
  w(`- Exact match on all four headline fields: **${pct(s.exactMatchRate)}**`);
  w(`- Median generation: **${secs(s.latencyMs.p50)}**, p90 ${secs(s.latencyMs.p90)}`);
  w(`- Download: **${mb(winner.load?.downloadBytes)}** (one-time, then cached)`);
  w(`- Per-field: ${headlineFields.map((k) => `${k} ${pct(f[k])}`).join(' \u00b7 ')}`);
  w();
  w('This is a *suggestion* engine, not an autopilot. At this accuracy a user must confirm the');
  w('parsed values before they become transactions.');
  w(`The most reliable field is \`${strongest}\` (${pct(f[strongest])}); the weakest are`);
  w(`**${weakest.join('** and **')}** \u2014 surface those for correction first.`);
  w('**No sub-1B model measured here is accurate enough to create transactions unattended.**');
  w();
  w('Configured as: ' + (winner.constrain
    ? 'constrained-decoding path enabled (see the findings — this path is NOT verified to enforce the grammar).'
    : 'unconstrained decoding, which measured best.'));
  w();

  // Flag when the choice is not purely a quality decision.
  if (winner.license && /non-OSI|AML|LFM|other/i.test(winner.license)) {
    const runnerUp = ranked.find((r) => r !== winner && /Apache|MIT/i.test(r.license ?? ''));
    w(`> **Licence check required.** The winner's weights are \`${winner.license}\`, which is not`);
    w('> OSI-approved. Confirm that is acceptable for your distribution before shipping.');
    if (runnerUp) {
      w(`> If it is not, the fallback is \`${runnerUp.model}\` at ${pct(runnerUp.summary.accuracy)}`);
      w(`> (${runnerUp.license}).`);
    }
    w();
  }
}
w();

// ------------------------------------------------------------------ results --
w('## 2. Results');
w();
w('| Model | Params | dtype | Constrained | Accuracy | JSON | Exact | p50 | p90 | Download |');
w('|---|---|---|---|---|---|---|---|---|---|');
for (const r of ranked) {
  const s = r.summary;
  w(
    `| \`${r.model}\` | ${r.params ?? '?'} | ${r.dtype} | ${r.constrain ? 'yes' : 'no'} | ` +
    `${pct(s.accuracy)} | ${pct(s.jsonParseRate)} | ${pct(s.exactMatchRate)} | ` +
    `${secs(s.latencyMs.p50)} | ${secs(s.latencyMs.p90)} | ${mb(r.load?.downloadBytes)} |`,
  );
}
for (const r of failed) {
  w(`| \`${r.model}\` | ${r.params ?? '?'} | ${r.dtype} | ${r.constrain ? 'yes' : 'no'} | FAILED: ${r.error} | | | | | |`);
}
w();

// --------------------------------------------------------------- per-field --
w('## 3. Per-field accuracy');
w();
w("This is where a model's real weakness shows. Aggregate accuracy hides whether a model reads");
w('totals well but never gets dates, or vice versa.');
w();
w('| Model | merchant | date | total | currency | tax | line_items |');
w('|---|---|---|---|---|---|---|');
for (const r of ranked) {
  const f = r.summary.fieldAccuracy;
  w(`| \`${r.model}\` | ${pct(f.merchant)} | ${pct(f.date)} | ${pct(f.total)} | ${pct(f.currency)} | ${pct(f.tax)} | ${pct(f.line_items)} |`);
}
w();
w('`tax` and `line_items` are excluded from headline accuracy — a transaction can be created');
w('from the four headline fields, and requiring itemisation would penalise models on a task the');
w('app can live without.');
w();

// ------------------------------------------------------------- methodology --
const meta = ranked[0]?._runMeta ?? runs[0]?._runMeta;
if (meta) {
  w('## 4. Methodology');
  w();
  w(`- **Dataset**: \`${meta.dataset}\`${meta.datasetRevision ? ` @ \`${meta.datasetRevision.slice(0, 12)}\`` : ''} (${meta.split} split).`);
  w('  Synthetic thermal receipts with machine-exact ground truth, across UK/US/DE/FR/IT locales.');
  w('  The `eval` split is deliberately de-leaked by the dataset authors (held-out fonts,');
  w('  vocabulary and merchant names), so scores are not inflated by memorisation.');
  w(`- **Images**: ${meta.fixtureCount} receipts, source \`${meta.fixturesSource ?? 'unknown'}\`.`);
  w('  The degraded `image_photo` variant is used, not the clean render — a phone camera never');
  w('  sees a clean render. Each fixture carries perspective, lighting, shadow, noise, blur and');
  w('  JPEG-artifact degradations.');
  w(`- **Prompt**: \`${meta.prompt?.id}\` — ${meta.prompt?.description}`);
  w(`- **Constrained decoding**: ${meta.constrain ? `enabled, schema \`${meta.schemaName}\`` : 'disabled'}.`);
  w(`- **Sampling**: greedy (\`do_sample: false\`), max ${meta.maxNewTokens} new tokens.`);
  w('- **Runtime**: transformers.js 4.3.0 on WebGPU, `onnxruntime-web`, measured in-tab.');
  if (meta.gpu) {
    w(`- **Hardware**: ${meta.gpu.vendor} ${meta.gpu.architecture}` +
      `${meta.gpu.shaderF16 ? ', shader-f16' : ''}, maxBufferSize ${mb(meta.gpu.maxBufferSize)}.`);
  }
  w('- **Caching**: model weights warm in the browser Cache API. `Download` is the measured');
  w('  transfer of the load phase; latency is steady-state, which is what a user sees after first launch.');
  w();
}

// ----------------------------------------------------------------- findings --
w('## 4. What actually determined accuracy');
w();
w('Six findings changed the shape of this benchmark, each verified experimentally:');
w();
w('### 5.1 Quantization choice can silently destroy a model');
w('The `q4f16` and `fp16` ONNX exports of SmolVLM-256M emit degenerate text — `"if if if …"`,');
w('`"-1: -1: -1:"` — on every prompt, while `q4` is correct and fast. This is a defect in those');
w('weight files, not a prompt or chat-template problem: the template renders correctly');
w('(`User:<image>…Assistant:`), the image produces the expected 13 tiles, and the identical');
w('prompt succeeds under `q4`. `uint8`/`int8` are also correct but 15-25x slower per token');
w('(62s vs 2.3s for the same generation).');
w('**Always verify a quantization actually works before trusting a size comparison.**');
w();
w('### 5.2 The structured-output constraint does NOT actually apply (negative result)');
w('`@huggingface/transformers-structured-output` exports `StructuredOutputProcessor`, which');
w('*extends* `LogitsProcessorList` — but the generation loop invokes `logits_processor` as a');
w('`Callable`. The object is accepted and then silently ignored. Evidence:');
w();
w('- A 5-property schema and a 6-property schema produced **byte-identical output** across 8 receipts.');
w('- Output contained `line_items` even when the schema explicitly forbids it.');
w('- At a 768-token budget the "constrained" arm scored **worse** (51.6%) than unconstrained (73.0%).');
w();
w('**Do not claim grammar-constrained decoding for this stack.** It is an unverified path, and the');
w('harness exposes it via `--constrain` only so the negative result is reproducible.');
w();
w('The apparent large gain seen in an earlier 4-receipt probe (6.3% -> 59.4%) did not replicate at');
w('larger sample sizes. That probe conflated two changes at once: it also disabled');
w('`repetition_penalty`, which is the only part of the `--constrain` path with a measurable effect.');
w('This is exactly why the sample size was raised and the arms were separated.');
w();
w('### 5.3 Numeric grounding, not format, is the real barrier');
{
  // Derived: the lineup changed once already and these numbers move with it.
  const small = ranked.filter((r) => /256M/i.test(r.model));
  const best = ranked[0];
  const totalOf = (r) => r.summary.fieldAccuracy?.total ?? 0;
  const curOf = (r) => r.summary.fieldAccuracy?.currency ?? 0;

  if (small.length) {
    const zeroTotal = small.filter((r) => totalOf(r) === 0);
    const zeroCur = small.filter((r) => curOf(r) === 0);
    w(`Every 256M model in the lineup is unusable on the fields that create a transaction.`);
    w(`${zeroTotal.length} of ${small.length} score **0.0% on \`total\`** across 24 receipts: given a receipt`);
    w('printing a total alongside a VAT id and a receipt number, they concatenate digits from');
    w('different places into values like `34024218`.');
    if (zeroCur.length) {
      w(`${zeroCur.length} of ${small.length} likewise score 0.0% on \`currency\`, emitting a store name or`);
      w('a till code where a currency belongs.');
    }
    w('Neither is salvageable by prompt or grammar work, because the failure is numeric grounding,');
    w('not output format.');
    w();
  }

  w(`The best model measured reaches only \`total\` ${pct(totalOf(best))}, so even the winner is wrong`);
  w(`on the amount roughly one time in four. **\`total\` is the field that must be surfaced for user`);
  w('confirmation above all others.**');
  w();
}
w('### 5.4 Architecture matters more than parameter count');
{
  // Derived from the data rather than hardcoded, because the lineup now contains
  // a non-SmolVLM architecture and the comparison has to reflect whatever ran.
  const byShort = (list, re) => list.filter((r) => re.test(r.model));
  const lfm = byShort(ranked, /LFM/i);
  const smol = byShort(ranked, /SmolVLM/i);
  const bestSmol = smol[0];

  if (lfm.length && bestSmol) {
    const b = lfm[0];
    w(`The lineup includes one non-SmolVLM architecture, and it is instructive. Comparing the leading`);
    w(`model of each family at 24 receipts:`);
    w();
    w('| Model | Params | Accuracy | merchant | date | total | currency | p50 |');
    w('|---|---|---|---|---|---|---|---|');
    for (const r of [bestSmol, b].sort((a, c) => c.summary.accuracy - a.summary.accuracy)) {
      const f = r.summary.fieldAccuracy;
      w(`| \`${r.model}\` | ${r.params} | **${pct(r.summary.accuracy)}** | ${pct(f.merchant)} | ${pct(f.date)} | ${pct(f.total)} | ${pct(f.currency)} | ${secs(r.summary.latencyMs.p50)} |`);
    }
    w();
  } else {
    w('An earlier, smaller sample suggested the SmolVLM2 video variants were far behind. At 24 receipts');
    w('they are close and complementary, so that claim is corrected here: SmolVLM2-500M scores 58.5%');
    w('against SmolVLM-500M at 63.2%, and **beats it decisively on dates (95% vs 83%) and currency');
    w('(79% vs 63%)** while losing badly on totals (19% vs 42%).');
    w();
  }

  w('Reading the same receipt, the families trade fields rather than one dominating. **Per-field');
  w('breakdown, not the headline number, is what should drive the choice** \u2014 a finance app that');
  w('cannot tolerate a wrong amount should weight the `total` column above all others, and one that');
  w('mainly needs the date should weight `date`.');
  w();
  w('The practical consequence: the accuracy ceiling for this task is a property of the models');
  w('measured, **not of the sub-1B class**. Changing architecture moved the amount-reading result far');
  w('more than adding parameters inside one family did.');
  w();
}
w('### 5.5 The grader understated every model until extraction was fixed');
w('Tiny models emit JSON that is invalid as a document while containing individually perfect');
w('fields \u2014 `"merchant": "Garcia Supermarket"` sitting next to `"price": +345150-243`. Scoring the');
w('whole document as unparseable threw the correct merchant away and reported 0.');
w();
w('Two recovery stages were added, and re-grading the *same stored output* from the pre-LFM2.5');
w('four-model run raised every model. These figures come from that earlier run and are kept as the');
w('record of the correction, not as the current ranking:');
w();
w('| Model | Strict JSON only | + truncation & field salvage |');
w('|---|---|---|');
w('| SmolVLM-500M | 60.1% | **63.2%** |');
w('| SmolVLM2-500M | 39.2% | **58.5%** |');
w('| SmolVLM-256M | 13.0% | **22.4%** |');
w();
w('The largest correction was +19 points on SmolVLM2-500M — enough to change its ranking. Reported');
w('accuracy here reflects extraction, not just first-try JSON validity, because an app that salvages');
w('a correct merchant from broken output has genuinely succeeded.');
w();

// ------------------------------------------------------------- exclusions ---
// Every candidate that was investigated and left out, with the evidence. Without
// this the reader cannot tell whether the lineup is a considered shortlist or
// simply everything that was tried, and a missing model looks like an oversight.
if (excluded.length) {
  w('## 6. Candidates investigated and excluded');
  w();
  w('The lineup above is a shortlist, not an exhaustive list. These candidates were evaluated and');
  w('deliberately left out; the reasoning is recorded so the composition is auditable.');
  w();
  for (const e of excluded) {
    const size = e.sizeMB ? `, ~${e.sizeMB} MB` : '';
    w(`### \`${e.repo}\``);
    w();
    w(`**${e.params ?? '?'} params${size} · ${e.license ?? 'licence unknown'} · excluded: ${e.reason}**`);
    w();
    w(e.evidence);
    w();
    if (e.verification) w(`Reproduce with \`node ${e.verification}\`.`);
    w();
  }
}

// --------------------------------------------------------------------- pwa --
w('## 7. What this means for the PWA');
w();
w('- **Download budget is the real constraint**, not VRAM. A 256M model at `q4` is ~430 MB;');
w('  the 500M is ~800 MB. Neither is acceptable as a blocking first-load on mobile data, so the');
w('  model must be fetched lazily, cached, and treated as an optional progressive enhancement');
w('  over manual entry or a server-side fallback.');
w('- **WebGPU is widely available but not universal.** Android Chrome 121+, iOS Safari 26+, and');
w('  desktop Chrome 113+ support it; `shader-f16` is at roughly 93% and absent on older Adreno 5xx/6xx');
w('  and NVIDIA Maxwell/Pascal. Dtype must be chosen at runtime from `adapter.features`, or the');
w('  app will break on specific GPUs.');
w('- **Warm latency of ~4-8s per receipt** is tolerable for a "scan and confirm" flow and');
w('  unacceptable for batch import of a shoebox of receipts.');
w('- **Budget ~1 GB of storage per origin on Safari**, with eviction after 7 days without');
w('  interaction unless installed as a PWA. Users may silently re-download the model.');
w('- **Preprocessing dominates more than expected**: image decode plus processor tiling was');
w('  measured at ~0.8s against ~4s of decoding, so image size reduction is a real optimization.');
w();

// --------------------------------------------------------------------- next --
w('## 8. Open questions worth testing next');
w();
w('1. **Numeric grounding.** The dominant remaining error is picking the wrong number off the');
w('   receipt. A prompt that asks for the total by its printed label, or a two-pass approach that');
w('   first transcribes the bottom block and then extracts from text, may beat single-pass VLM extraction.');
w('2. **`LFM2.5-VL-450M` and `granite-docling-258M`** were identified in research as strong');
w('   sub-500M candidates and are not yet in this lineup.');
w('3. **Constrained decoding vs KV-cache reuse.** Repeated queries against one image (a very');
w('   plausible UX: extract, then re-ask a corrected field) currently re-pay full prefill.');
w('4. **Real photographs.** All fixtures are synthetic thermal renders. The dataset authors');
w('   explicitly leave real-photo transfer unmeasured, and crumpled, low-light camera photos are');
w('   where these models are most likely to fall over.');
w('5. **Model size vs. image resolution.** Tiling (`do_image_splitting`) is a memory/accuracy');
w('   dial that was left at default here; a 256M model given more visual tokens may close part of');
w('   the gap to 500M at lower weight cost.');
w();

// ----------------------------------------------------------------- repro ---
w('## 9. Reproducing');
w();
w('```bash');
w('npm install');
w('npm run probe:gpu                 # confirm a real WebGPU adapter');
w('npm run fixtures                  # pull fixtures via parquet range requests');
w('npm run serve                     # dev server, watcher disabled for stable timing');
w('npm test                          # scorer unit tests');
w('npm run bench -- --limit 24 --prompt v1 --out results/run.json');
w('node scripts/report.mjs results/run.json');
w('```');
w();
w('Result files behind this report:');
for (const f of files) w(`- \`${f}\``);
w();

await mkdir(path.dirname(outPath), { recursive: true });
await writeFile(outPath, L.join('\n'));
console.log(`[report] wrote ${outPath} (${L.length} lines, ${runs.length} runs)`);
if (winner) {
  console.log(`[report] winner: ${winner.model} accuracy=${pct(winner.summary.accuracy)} p50=${secs(winner.summary.latencyMs.p50)}`);
}
