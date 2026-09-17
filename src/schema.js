/**
 * The output contract every candidate model is graded against.
 *
 * This mirrors what a personal-finance app actually needs in order to create a
 * transaction the user won't have to hand-correct: who, when, how much, in what
 * currency, plus the optional breakdown.
 *
 * Deliberately independent of any model or runtime so the same scorer grades
 * every candidate — and so it can be unit-tested in plain Node.
 */

/** Canonical transaction shape. */
export const FIELDS = ['merchant', 'date', 'total', 'currency', 'tax', 'line_items'];

/** Fields weighted into the headline accuracy score. `tax`/`line_items` are bonus. */
export const SCORED_FIELDS = ['merchant', 'date', 'total', 'currency'];

export const DATE_PRECISION = {
  EXACT: 1.0,
  MONTH_DAY: 0.8, // year wrong/missing but month+day right
  MONTH_YEAR: 0.5, // day wrong/missing
  YEAR_ONLY: 0.2,
  NONE: 0,
};

/** ISO-4217 codes we expect to see on receipts. Used for currency validation. */
export const CURRENCY_SYMBOLS = {
  $: 'USD',
  'US$': 'USD',
  '€': 'EUR',
  '£': 'GBP',
  '¥': 'JPY',
  '₹': 'INR',
  C$: 'CAD',
  A$: 'AUD',
  CHF: 'CHF',
  kr: 'SEK',
  zł: 'PLN',
  R$: 'BRL',
  MX$: 'MXN',
};

const CURRENCY_CODES = new Set([
  'USD', 'EUR', 'GBP', 'JPY', 'INR', 'CAD', 'AUD', 'CHF', 'SEK', 'NOK', 'DKK',
  'PLN', 'CZK', 'HUF', 'RON', 'BRL', 'MXN', 'ARS', 'CLP', 'COP', 'PEN', 'ZAR',
  'TRY', 'AED', 'SAR', 'ILS', 'EGP', 'NGN', 'KES', 'GHS', 'CNY', 'HKD', 'TWD',
  'KRW', 'SGD', 'MYR', 'THB', 'IDR', 'PHP', 'VND', 'NZD', 'RUB', 'UAH', 'PKR',
  'BDT', 'LKR', 'MAD', 'TND', 'DZD', 'QAR', 'KWD', 'BHD', 'OMR', 'JOD', 'LBP',
]);

/**
 * Pull the first JSON object out of raw model text.
 *
 * Tiny VLMs are reliably sloppy here: they wrap output in markdown fences, add a
 * "Here is the JSON:" preamble, emit trailing commentary, or leave trailing
 * commas. This handles all of those without resorting to eval().
 *
 * @param {string} text
 * @returns {{ value: any, strategy: string } | { value: null, strategy: string, error: string }}
 */
export function extractJson(text) {
  if (typeof text !== 'string' || !text.trim()) {
    return { value: null, strategy: 'empty-input', error: 'no text to parse' };
  }

  // 1. Direct parse.
  const trimmed = text.trim();
  try {
    return { value: JSON.parse(trimmed), strategy: 'direct' };
  } catch { /* fall through */ }

  // 2. Strip markdown code fences.
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      return { value: JSON.parse(fenced[1].trim()), strategy: 'markdown-fence' };
    } catch { /* fall through */ }
  }

  // 3. Scan for the first balanced {...} block, respecting strings and escapes.
  const start = trimmed.indexOf('{');
  if (start !== -1) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < trimmed.length; i++) {
      const ch = trimmed[i];
      if (escaped) { escaped = false; continue; }
      if (ch === '\\') { if (inString) escaped = true; continue; }
      if (ch === '"') { inString = !inString; continue; }
      if (inString) continue;
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) {
          const candidate = trimmed.slice(start, i + 1);
          try {
            return { value: JSON.parse(candidate), strategy: 'balanced-scan' };
          } catch { /* keep scanning: try later braces */ }
        }
      }
    }
  }

  // 4. Last resort: repair the common tiny-model breakages and retry.
  const repaired = tryRepair(trimmed, fenced ? fenced[1] : trimmed);
  if (repaired !== null) {
    return { value: repaired, strategy: 'repaired' };
  }

  // 5. Output was cut off mid-object (hit the token ceiling). Closing the open
  //    brackets recovers every field emitted before the cut, which is the
  //    difference between scoring a good answer 0.0 and scoring it honestly.
  const salvaged = salvageTruncated(fenced ? fenced[1] : trimmed);
  if (salvaged !== null) {
    return { value: salvaged, strategy: 'truncated-salvage' };
  }

  // 6. Structurally invalid JSON, but with individually well-formed fields.
  //    Tiny models emit things like `"price": +345150-243` and unbalanced
  //    nesting that no bracket-closing can fix. Discarding a correct
  //    `"merchant": "Garcia Supermarket"` sitting next to the damage would
  //    understate the model, so known keys are pulled out individually.
  const fields = salvageFields(fenced ? fenced[1] : trimmed);
  if (fields !== null) {
    return { value: fields, strategy: 'field-salvage' };
  }

  return { value: null, strategy: 'unparseable', error: 'no valid JSON object found' };
}

/**
 * Last-resort per-field extraction from broken JSON.
 *
 * Only the contract's own keys are considered, and each value must still parse
 * as a clean string/number/null. This deliberately recovers a *subset*: a field
 * that cannot be read confidently stays absent rather than being guessed, so the
 * score reflects what the model really produced.
 *
 * @param {string} text
 * @returns {Record<string, any> | null}
 */
export function salvageFields(text) {
  if (typeof text !== 'string') return null;
  const out = {};

  for (const key of ['merchant', 'date', 'currency']) {
    const m = text.match(new RegExp(`"${key}"\\s*:\\s*"([^"\\\\]{1,120})"`, 'i'));
    if (m) out[key] = m[1].trim();
  }

  for (const key of ['total', 'tax']) {
    // Accept a plain number, or a quoted numeric-ish string. Reject fragments
    // like "+345150-243" by requiring a single well-formed numeric literal.
    const quoted = text.match(new RegExp(`"${key}"\\s*:\\s*"([^"\\\\]{1,40})"`, 'i'));
    const bare = text.match(new RegExp(`"${key}"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)\\s*[,}\\]\\n]`, 'i'));
    const raw = bare ? bare[1] : quoted ? quoted[1] : null;
    if (raw != null) {
      const n = parseAmount(raw);
      if (n != null) out[key] = n;
    }
  }

  return Object.keys(out).length ? out : null;
}

/**
 * Close a truncated JSON object and parse it.
 *
 * Truncation is extremely common with constrained decoding: the grammar keeps
 * producing valid tokens until `max_new_tokens` runs out mid-array. Anything
 * already emitted is real signal, so it is worth recovering rather than
 * discarding. Returns null if the fragment cannot be completed into valid JSON.
 *
 * @param {string} text
 * @returns {any | null}
 */
export function salvageTruncated(text) {
  if (typeof text !== 'string') return null;
  const start = text.indexOf('{');
  if (start === -1) return null;

  let body = text.slice(start);
  // Drop a dangling key ("... , \"amount\"") before attempting to close.
  body = body.replace(/,\s*"[^"]*"\s*:?\s*$/, '').replace(/,\s*$/, '');

  // Close whatever is still open, innermost first.
  const stack = [];
  let inString = false;
  let escaped = false;
  for (const ch of body) {
    if (escaped) { escaped = false; continue; }
    if (ch === '\\') { if (inString) escaped = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === '{' || ch === '[') stack.push(ch);
    else if (ch === '}' || ch === ']') stack.pop();
  }
  if (!stack.length && !inString) return null; // was not truncated

  const closers = { '{': '}', '[': ']' };
  const tails = [];
  if (inString) tails.push('"');
  for (let i = stack.length - 1; i >= 0; i--) tails.push(closers[stack[i]]);

  // Dangling commas before a closer are invalid JSON; try both with and without
  // stripping them, since the truncation point varies.
  for (const tail of [tails.join(''), tails.join('').replace(/^\}/, '}')]) {
    for (const candidate of [
      body + tail,
      body.replace(/,\s*$/, '') + tail,
      body.replace(/,\s*$/, '') + tail.replace(/([}\]])$/, ',$1'),
    ]) {
      try {
        const value = JSON.parse(candidate);
        if (value && typeof value === 'object') return value;
      } catch { /* try the next closing strategy */ }
    }
  }
  return null;
}

/** Best-effort repair of near-JSON. Returns null if it still won't parse. */
function tryRepair(original, fencedBody) {
  let body = fencedBody.trim();
  const first = body.indexOf('{');
  const last = body.lastIndexOf('}');
  if (first === -1) return null;
  body = last > first ? body.slice(first, last + 1) : body.slice(first);

  const stripTrailingCommas = (s) => s.replace(/,\s*([}\]])/g, '$1');
  const fixLiterals = (s) =>
    s.replace(/\bNone\b/g, 'null').replace(/\bTrue\b/g, 'true').replace(/\bFalse\b/g, 'false');
  const fixQuotes = (s) => s.replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, '"$1"');

  const variants = [
    // Trailing commas before a closing brace/bracket.
    stripTrailingCommas(body),
    // Python/JS literals.
    fixLiterals(stripTrailingCommas(body)),
    // Single-quoted keys and values -> double quotes.
    fixQuotes(stripTrailingCommas(body)),
    // Everything at once: tiny models routinely produce several of these.
    fixQuotes(fixLiterals(stripTrailingCommas(body))),
  ];

  for (const v of variants) {
    try { return JSON.parse(v); } catch { /* next */ }
  }
  return null;
}

/** Coerce anything into a trimmed string, mapping nullish to ''. */
export function asString(v) {
  if (v == null) return '';
  if (typeof v === 'string') return v.trim();
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return '';
}

/**
 * Parse a monetary value out of whatever the model produced.
 * Handles "$1,234.56", "1234,56" (EU decimal comma), "12.34 EUR", numbers.
 * @returns {number | null} value in major units, or null if unparseable
 */
export function parseAmount(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  const s = asString(v);
  if (!s) return null;
  let cleaned = s.replace(/[^\d.,\-]/g, '');
  if (!cleaned) return null;
  const lastComma = cleaned.lastIndexOf(',');
  const lastDot = cleaned.lastIndexOf('.');
  if (lastComma > lastDot && lastComma !== -1) {
    // Decimal comma: treat '.' as thousands separator.
    cleaned = cleaned.replace(/\./g, '').replace(',', '.');
  } else {
    cleaned = cleaned.replace(/,/g, '');
  }
  const n = Number.parseFloat(cleaned);
  return Number.isFinite(n) ? n : null;
}

/**
 * Normalize a date to ISO YYYY-MM-DD.
 * Accepts ISO, US (MM/DD/YYYY), EU (DD/MM/YYYY), and "12 Mar 2024" style.
 *
 * Ambiguous numeric dates (both parts <= 12) cannot be resolved from the string
 * alone. The receipt's locale is real ground-truth signal — a German receipt's
 * "07.10.2021" is 7 October — so callers that know it should pass `locale`, and
 * we resolve accordingly while recording which assumption was used.
 *
 * @param {any} v
 * @param {{ locale?: string | null }} [opts]
 * @returns {{ iso: string | null, assume: string }}
 */
export function parseDate(v, opts = {}) {
  const s = asString(v);
  if (!s) return { iso: null, assume: 'empty' };

  // Already ISO.
  let m = s.match(/\b(\d{4})-(\d{2})-(\d{2})\b/);
  if (m) return { iso: `${m[1]}-${m[2]}-${m[3]}`, assume: 'iso' };

  const MONTHS = {
    jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6,
    jul: 7, aug: 8, sep: 9, sept: 9, oct: 10, nov: 11, dec: 12,
  };

  // "12 Mar 2024" / "Mar 12, 2024" / "12-Mar-24"
  m = s.match(/\b(\d{1,2})[\s\-\/.]*([A-Za-z]{3,9})[\s\-\/.,]*(\d{2,4})\b/);
  if (m && MONTHS[m[2].slice(0, 4).toLowerCase()]) {
    return { iso: iso(m[3], MONTHS[m[2].slice(0, 4).toLowerCase()], m[1]), assume: 'day-monthname-year' };
  }
  m = s.match(/\b([A-Za-z]{3,9})[\s\-\/.]*(\d{1,2})[\s\-\/.,]*(\d{2,4})\b/);
  if (m && MONTHS[m[1].slice(0, 4).toLowerCase()]) {
    return { iso: iso(m[3], MONTHS[m[1].slice(0, 4).toLowerCase()], m[2]), assume: 'monthname-day-year' };
  }

  // Numeric d/m/y or m/d/y.
  m = s.match(/\b(\d{1,2})[\/\-. ](\d{1,2})[\/\-. ](\d{2,4})\b/);
  if (m) {
    const a = Number(m[1]);
    const b = Number(m[2]);
    const year = normYear(m[3]);
    if (a > 12 && b <= 12) return { iso: iso(year, b, a), assume: 'day-first-forced' };
    if (b > 12 && a <= 12) return { iso: iso(year, a, b), assume: 'month-first-forced' };
    // Genuinely ambiguous: use locale if we have it.
    if (isDayFirstLocale(opts.locale)) return { iso: iso(year, b, a), assume: 'day-first-locale' };
    if (opts.locale) return { iso: iso(year, a, b), assume: 'month-first-locale' };
    return { iso: iso(year, a, b), assume: 'month-first-ambiguous' };
  }

  return { iso: null, assume: 'unrecognized' };
}

/** Locales in the benchmark corpus that write the day before the month. */
function isDayFirstLocale(locale) {
  if (!locale) return false;
  return !/^(US|CA|PH|MX)$/i.test(String(locale));
}

function normYear(y) {
  const n = Number(y);
  if (y.length === 2) return String(n >= 70 ? 1900 + n : 2000 + n);
  return String(n).padStart(4, '0');
}

function iso(y, m, d) {
  const yy = normYear(String(y));
  const mm = String(m).padStart(2, '0');
  const dd = String(d).padStart(2, '0');
  return `${yy}-${mm}-${dd}`;
}

/**
 * Normalize a currency to an ISO code.
 * @returns {string | null}
 */
export function parseCurrency(v) {
  const raw = asString(v);
  if (!raw) return null;
  const upper = raw.toUpperCase().replace(/[^A-Z$€£¥₹]/g, '');
  if (CURRENCY_CODES.has(upper)) return upper;
  for (const [sym, code] of Object.entries(CURRENCY_SYMBOLS)) {
    if (raw.includes(sym)) return code;
  }
  if (/^[A-Z]{3}$/.test(upper) && CURRENCY_CODES.has(upper)) return upper;
  return null;
}

/** Normalize merchant names for comparison: case, punctuation, common noise. */
export function normalizeMerchant(v) {
  return asString(v)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\b(inc|llc|ltd|limited|corp|corporation|co|gmbh|sarl|bv|nv|plc|store|branch|#\d+)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Token-overlap F1 between two normalized merchant names. */
export function merchantSimilarity(a, b) {
  const ta = normalizeMerchant(a).split(' ').filter(Boolean);
  const tb = normalizeMerchant(b).split(' ').filter(Boolean);
  if (!ta.length && !tb.length) return 1;
  if (!ta.length || !tb.length) return 0;
  const setB = new Set(tb);
  let overlap = 0;
  for (const t of new Set(ta)) if (setB.has(t)) overlap++;
  const precision = overlap / new Set(ta).size;
  const recall = overlap / setB.size;
  if (precision + recall === 0) return 0;
  return (2 * precision * recall) / (precision + recall);
}

/**
 * Score a single field. Returns a value in [0,1].
 * @param {string} field
 * @param {any} expected
 * @param {any} actual
 * @param {{ locale?: string | null }} [opts] locale disambiguates numeric dates
 */
export function scoreField(field, expected, actual, opts = {}) {
  switch (field) {
    case 'merchant': {
      const e = asString(expected);
      const a = asString(actual);
      if (!e) return 1; // nothing to get wrong
      if (!a) return 0;
      const sim = merchantSimilarity(e, a);
      // Exact-normalized match is a clean hit; near matches earn partial credit.
      if (normalizeMerchant(e) === normalizeMerchant(a)) return 1;
      return sim >= 0.5 ? 0.5 + (sim - 0.5) : 0;
    }
    case 'date': {
      const e = parseDate(expected, opts).iso;
      const a = parseDate(actual, opts).iso;
      if (!e) return 1;
      if (!a) return 0;
      if (e === a) return 1;
      const [ey, em, ed] = e.split('-');
      const [ay, am, ad] = a.split('-');
      if (ey === ay && em === am && ed === ad) return 1;
      if (em === am && ed === ad && ey !== ay) return DATE_PRECISION.MONTH_DAY;
      if (ey === ay && em === am) return DATE_PRECISION.MONTH_YEAR;
      if (ey === ay) return DATE_PRECISION.YEAR_ONLY;
      return 0;
    }
    case 'total': {
      const e = parseAmount(expected);
      const a = parseAmount(actual);
      if (e == null) return 1;
      if (a == null) return 0;
      const diff = Math.abs(e - a);
      if (diff < 0.005) return 1; // exact to the cent
      if (diff <= 0.02) return 0.9; // rounding slop
      if (e !== 0 && diff / Math.abs(e) <= 0.01) return 0.5; // within 1%
      return 0;
    }
    case 'currency': {
      const e = parseCurrency(expected);
      const a = parseCurrency(actual);
      if (!e) return 1;
      if (!a) return 0;
      return e === a ? 1 : 0;
    }
    case 'tax': {
      const e = parseAmount(expected);
      const a = parseAmount(actual);
      if (e == null) return 1;
      if (a == null) return 0;
      const diff = Math.abs(e - a);
      if (diff < 0.005) return 1;
      if (diff <= 0.02) return 0.9;
      return 0;
    }
    case 'line_items': {
      const e = Array.isArray(expected) ? expected : [];
      const a = Array.isArray(actual) ? actual : [];
      if (!e.length) return 1;
      if (!a.length) return 0;
      // Greedy match by description similarity, then compare amounts.
      const used = new Set();
      let matched = 0;
      let amountOk = 0;
      for (const exp of e) {
        let bestIdx = -1;
        let bestSim = 0;
        a.forEach((act, i) => {
          if (used.has(i)) return;
          const sim = merchantSimilarity(
            exp?.description ?? exp?.name ?? '',
            act?.description ?? act?.name ?? act?.item ?? '',
          );
          if (sim > bestSim) { bestSim = sim; bestIdx = i; }
        });
        if (bestIdx >= 0 && bestSim >= 0.5) {
          used.add(bestIdx);
          matched++;
          const ea = parseAmount(exp?.amount ?? exp?.total ?? exp?.price);
          const aa = parseAmount(a[bestIdx]?.amount ?? a[bestIdx]?.total ?? a[bestIdx]?.price);
          if (ea != null && aa != null && Math.abs(ea - aa) < 0.005) amountOk++;
        }
      }
      const descF1 = matched / Math.max(e.length, a.length);
      const amtF1 = amountOk / Math.max(e.length, a.length);
      return 0.5 * descF1 + 0.5 * amtF1;
    }
    default:
      return 0;
  }
}

/**
 * Grade a model's raw output against ground truth.
 *
 * `fieldScores` always covers every field so we can report per-field weakness
 * (e.g. "reads totals fine, never gets dates right"), while `accuracy` is the
 * mean over SCORED_FIELDS only.
 */
export function scoreReceipt({ raw, parsed, expected, parseStrategy, locale }) {
  const got = parsed && typeof parsed === 'object' ? parsed : {};
  const opts = { locale: locale ?? expected?.locale ?? null };
  const fieldScores = {};
  for (const f of FIELDS) {
    fieldScores[f] = scoreField(f, expected?.[f] ?? null, got[f] ?? null, opts);
  }
  const scored = SCORED_FIELDS.map((f) => fieldScores[f]);
  const accuracy = scored.length ? scored.reduce((a, b) => a + b, 0) / scored.length : 0;
  const allFields = FIELDS.map((f) => fieldScores[f]);
  const accuracyAllFields = allFields.reduce((a, b) => a + b, 0) / allFields.length;
  return {
    accuracy,
    accuracyAllFields,
    fieldScores,
    exactMatch: scored.every((s) => s === 1),
    parsedOk: !!parsed && typeof parsed === 'object',
    parseStrategy: parseStrategy ?? null,
    rawLength: typeof raw === 'string' ? raw.length : 0,
  };
}

/** Aggregate many per-receipt results into a model-level summary. */
export function summarize(results) {
  const n = results.length || 1;
  const mean = (fn) => results.reduce((a, r) => a + fn(r), 0) / n;
  const fieldMeans = {};
  for (const f of FIELDS) {
    fieldMeans[f] = results.reduce((a, r) => a + (r.score?.fieldScores?.[f] ?? 0), 0) / n;
  }
  const latencies = results.map((r) => r.metrics?.totalMs ?? 0).filter((x) => x > 0).sort((a, b) => a - b);
  const pct = (p) => (latencies.length ? latencies[Math.min(latencies.length - 1, Math.floor(p * latencies.length))] : null);
  const parseStrategies = {};
  for (const r of results) {
    const s = r.score?.parseStrategy ?? 'none';
    parseStrategies[s] = (parseStrategies[s] ?? 0) + 1;
  }
  return {
    count: results.length,
    accuracy: mean((r) => r.score?.accuracy ?? 0),
    accuracyAllFields: mean((r) => r.score?.accuracyAllFields ?? 0),
    exactMatchRate: mean((r) => (r.score?.exactMatch ? 1 : 0)),
    jsonParseRate: mean((r) => (r.score?.parsedOk ? 1 : 0)),
    fieldAccuracy: fieldMeans,
    parseStrategies,
    latencyMs: {
      p50: pct(0.5),
      p90: pct(0.9),
      mean: latencies.length ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null,
      min: latencies[0] ?? null,
      max: latencies[latencies.length - 1] ?? null,
    },
    tokensPerSecond: mean((r) => r.metrics?.tokensPerSecond ?? 0) || null,
  };
}
