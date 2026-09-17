/**
 * Re-grade an existing result file with the current scorer.
 *
 * Model output is the expensive, durable artifact; scoring is cheap and evolves.
 * When the grader is fixed or tightened, existing runs should be re-scored rather
 * than re-run — re-running would also re-download models and burn GPU time.
 *
 *   node scripts/rescore.mjs results/bench-constrained.json [--out results/x.json]
 */
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { extractJson, scoreReceipt, summarize } from '../src/schema.js';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const input = process.argv.slice(2).find((a) => !a.startsWith('--') && a.endsWith('.json'));
if (!input) {
  console.error('usage: node scripts/rescore.mjs <results.json> [--out results/x.json]');
  process.exit(1);
}
const outPath = arg('out', input.replace(/\.json$/, '.rescored.json'));

const data = JSON.parse(await readFile(input, 'utf8'));
const split = data.split ?? 'eval';
const manifest = JSON.parse(await readFile(path.join('fixtures', split, 'manifest.json'), 'utf8'));
const byId = new Map(manifest.entries.map((e) => [e.id, e]));

let changed = 0;
const strategyShifts = {};

for (const run of data.runs ?? []) {
  if (!run.results?.length) continue;
  for (const r of run.results) {
    const entry = byId.get(r.id);
    if (!entry) continue;
    const expected = JSON.parse(await readFile(path.join(process.cwd(), entry.label), 'utf8'));
    const before = r.score?.accuracy ?? 0;
    const beforeStrategy = r.score?.parseStrategy;

    const parsed = r.ok === false ? { value: null, strategy: 'no-output' } : extractJson(r.text ?? '');
    r.score = scoreReceipt({
      raw: r.text ?? '',
      parsed: parsed.value,
      expected,
      parseStrategy: parsed.strategy,
      locale: entry.locale ?? r.meta?.locale,
    });
    r.parsed = parsed.value ?? null;
    r.meta = { ...(r.meta ?? {}), locale: entry.locale, degradations: entry.degradations, bytes: entry.bytes, nItems: entry.nItems };

    if (Math.abs(r.score.accuracy - before) > 1e-9) changed++;
    if (beforeStrategy !== parsed.strategy) {
      const key = `${beforeStrategy} -> ${parsed.strategy}`;
      strategyShifts[key] = (strategyShifts[key] ?? 0) + 1;
    }
  }
  run.summary = summarize(run.results);
}

data.rescoredAt = new Date().toISOString();
data.rescoreNote = 'Re-graded from stored model output with the current scorer; no inference was re-run.';

await writeFile(outPath, JSON.stringify(data, null, 2));

console.log(`[rescore] ${input} -> ${outPath}`);
console.log(`[rescore] ${changed} receipts changed score`);
if (Object.keys(strategyShifts).length) {
  console.log('[rescore] parse strategy shifts:');
  for (const [k, v] of Object.entries(strategyShifts).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${k}: ${v}`);
  }
}
console.log('');
for (const run of data.runs ?? []) {
  if (!run.summary) { console.log(`${run.model.padEnd(46)} FAILED`); continue; }
  const s = run.summary;
  console.log(
    `${run.model.padEnd(46)} acc=${(s.accuracy * 100).toFixed(1)}% json=${(s.jsonParseRate * 100).toFixed(0)}% ` +
    `exact=${(s.exactMatchRate * 100).toFixed(0)}% p50=${s.latencyMs.p50 ? (s.latencyMs.p50 / 1000).toFixed(1) + 's' : '--'}`,
  );
}
