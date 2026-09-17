/**
 * Unit tests for the grading core.
 *
 * These matter more than they look: every accuracy number in the benchmark is
 * produced by this file. A scorer that silently accepts wrong values would make
 * the whole model comparison meaningless, so the tricky cases (EU decimal
 * commas, ambiguous dates, tiny-model JSON damage, merchant name noise) are
 * pinned down here.
 *
 *   node --test tests/
 */
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  extractJson,
  parseAmount,
  parseDate,
  parseCurrency,
  normalizeMerchant,
  merchantSimilarity,
  scoreField,
  scoreReceipt,
  summarize,
} from '../src/schema.js';

describe('extractJson', () => {
  test('parses clean JSON', () => {
    const r = extractJson('{"total": 5}');
    assert.equal(r.value.total, 5);
    assert.equal(r.strategy, 'direct');
  });

  test('strips markdown fences', () => {
    const r = extractJson('```json\n{"total": 5}\n```');
    assert.equal(r.value.total, 5);
    assert.equal(r.strategy, 'markdown-fence');
  });

  test('recovers JSON from prose preamble and trailing commentary', () => {
    const r = extractJson('Here is the JSON you asked for:\n{"merchant": "ACME", "total": 12.5}\nLet me know if you need more!');
    assert.equal(r.value.merchant, 'ACME');
    assert.equal(r.value.total, 12.5);
  });

  test('repairs trailing commas', () => {
    const r = extractJson('{"a": 1, "b": 2,}');
    assert.equal(r.value.b, 2);
  });

  test('repairs Python literals', () => {
    const r = extractJson("{'total': None, 'paid': True}");
    assert.equal(r.value.total, null);
    assert.equal(r.value.paid, true);
  });

  test('handles braces inside string values without truncating', () => {
    const r = extractJson('preamble {"note": "use {braces} carefully", "total": 9} trailing');
    assert.equal(r.value.note, 'use {braces} carefully');
    assert.equal(r.value.total, 9);
  });

  test('handles escaped quotes inside strings', () => {
    const r = extractJson('{"name": "Bob\\"s Diner", "total": 3}');
    assert.equal(r.value.name, 'Bob"s Diner');
  });

  test('reports failure on no JSON', () => {
    const r = extractJson('I cannot read this receipt.');
    assert.equal(r.value, null);
    assert.equal(r.strategy, 'unparseable');
  });

  test('reports failure on empty input', () => {
    const r = extractJson('');
    assert.equal(r.value, null);
  });

  test('handles nested objects', () => {
    const r = extractJson('{"a": {"b": {"c": 1}}, "d": 2}');
    assert.equal(r.value.a.b.c, 1);
    assert.equal(r.value.d, 2);
  });

  test('salvages output truncated mid-array by the token ceiling', () => {
    // Reproduces a real SmolVLM-500M output that hit max_new_tokens inside
    // line_items. The scalar fields it emitted first are all correct and must
    // not be thrown away.
    const truncated =
      '{\n  "merchant": "Garcia Supermarket",\n  "date": "23/07/2020",\n' +
      '  "total": 184.89,\n  "currency": "GBP",\n  "tax": 30.81,\n' +
      '  "line_items": [\n    {\n      "description": "Cobi Wrld Tanks",\n      "amount": 24.99\n    },\n' +
      '    {\n      "description": "Cellot 40P",\n      "amount": 29';
    const r = extractJson(truncated);
    assert.equal(r.strategy, 'truncated-salvage');
    assert.equal(r.value.merchant, 'Garcia Supermarket');
    assert.equal(r.value.total, 184.89);
    assert.equal(r.value.currency, 'GBP');
    assert.equal(r.value.tax, 30.81);
  });

  test('salvages truncation at a dangling key', () => {
    const r = extractJson('{"total": 10, "currency": "USD", "merchant"');
    assert.equal(r.strategy, 'truncated-salvage');
    assert.equal(r.value.total, 10);
    assert.equal(r.value.currency, 'USD');
  });

  test('does not fabricate a result from irrecoverable text', () => {
    const r = extractJson('the model rambled without ever opening a brace');
    assert.equal(r.value, null);
    assert.equal(r.strategy, 'unparseable');
  });
});

describe('parseAmount', () => {
  test('accepts plain numbers', () => {
    assert.equal(parseAmount(12.5), 12.5);
  });

  test('strips currency symbols and thousands separators', () => {
    assert.equal(parseAmount('$1,234.56'), 1234.56);
    assert.equal(parseAmount('£12.00'), 12);
    assert.equal(parseAmount('12.34 EUR'), 12.34);
  });

  test('handles European decimal comma', () => {
    assert.equal(parseAmount('1.234,56'), 1234.56);
    assert.equal(parseAmount('12,50'), 12.5);
  });

  test('rejects unparseable text', () => {
    assert.equal(parseAmount('n/a'), null);
    assert.equal(parseAmount(''), null);
    assert.equal(parseAmount(null), null);
  });
});

describe('parseDate', () => {
  test('passes through ISO', () => {
    assert.equal(parseDate('2020-07-23').iso, '2020-07-23');
  });

  test('parses UK day-first numeric date', () => {
    assert.equal(parseDate('23/07/2020').iso, '2020-07-23');
  });

  test('parses German dotted date as day-first when locale is known', () => {
    assert.equal(parseDate('07.10.2021', { locale: 'DE' }).iso, '2021-10-07');
    assert.equal(parseDate('07.10.2021', { locale: 'UK' }).iso, '2021-10-07');
  });

  test('resolves the same ambiguous string differently for US locale', () => {
    // 07.10.2021 is 7 October in DE but July 10 in the US.
    assert.equal(parseDate('07.10.2021', { locale: 'US' }).iso, '2021-07-10');
    assert.equal(parseDate('07.10.2021', { locale: 'DE' }).iso, '2021-10-07');
  });

  test('resolves month-first when day exceeds 12', () => {
    // 03/25/2021 can only be month-first.
    assert.equal(parseDate('03/25/2021').iso, '2021-03-25');
  });

  test('resolves day-first when first number exceeds 12', () => {
    assert.equal(parseDate('25/03/2021').iso, '2021-03-25');
  });

  test('flags genuinely ambiguous dates only when locale is unknown', () => {
    const r = parseDate('03/04/2021');
    assert.match(r.assume, /ambiguous|month-first/);
    // With a locale we commit to an interpretation rather than guessing.
    assert.equal(parseDate('03/04/2021', { locale: 'UK' }).assume, 'day-first-locale');
    assert.equal(parseDate('03/04/2021', { locale: 'US' }).assume, 'month-first-locale');
  });

  test('parses month names', () => {
    assert.equal(parseDate('12 Mar 2024').iso, '2024-03-12');
    assert.equal(parseDate('Mar 12, 2024').iso, '2024-03-12');
  });

  test('expands 2-digit years', () => {
    assert.equal(parseDate('15/01/19').iso, '2019-01-15');
    assert.equal(parseDate('15/01/99').iso, '1999-01-15');
  });

  test('returns null for garbage', () => {
    assert.equal(parseDate('sometime last week').iso, null);
  });
});

describe('parseCurrency', () => {
  test('accepts ISO codes', () => {
    assert.equal(parseCurrency('USD'), 'USD');
    assert.equal(parseCurrency('gbp'), 'GBP');
  });

  test('maps symbols', () => {
    assert.equal(parseCurrency('$'), 'USD');
    assert.equal(parseCurrency('€'), 'EUR');
    assert.equal(parseCurrency('£'), 'GBP');
  });

  test('rejects nonsense', () => {
    assert.equal(parseCurrency('dollars'), null);
    assert.equal(parseCurrency(''), null);
  });
});

describe('merchant normalization', () => {
  test('strips punctuation, case, and legal suffixes', () => {
    assert.equal(normalizeMerchant('ACME STORES, INC.'), 'acme stores');
    assert.equal(normalizeMerchant('The Corner Cafe Ltd'), 'the corner cafe');
  });

  test('similarity is high for near matches', () => {
    assert.equal(merchantSimilarity('GARCIA SUPERMARKET', 'Garcia Supermarket'), 1);
    // A one-letter OCR slip keeps half the tokens identical; that is the
    // documented partial-credit boundary.
    assert.ok(merchantSimilarity('GARCIA SUPERMARKET', 'GARCIA SUPERMARKT') >= 0.5);
  });

  test('similarity is low for unrelated names', () => {
    assert.equal(merchantSimilarity('ACME', 'Zebra Books'), 0);
  });
});

describe('scoreField', () => {
  test('merchant: exact and near matches score, wrong does not', () => {
    assert.equal(scoreField('merchant', 'ACME STORES', 'ACME STORES'), 1);
    assert.ok(scoreField('merchant', 'ACME STORES', 'ACME STORE') >= 0.5);
    assert.equal(scoreField('merchant', 'ACME STORES', 'Zebra Books'), 0);
    assert.equal(scoreField('merchant', 'ACME STORES', null), 0);
  });

  test('date: partial credit for partially-correct dates', () => {
    assert.equal(scoreField('date', '2020-07-23', '2020-07-23'), 1);
    assert.equal(scoreField('date', '2020-07-23', '23/07/2020'), 1);
    assert.ok(scoreField('date', '2020-07-23', '2020-07-01') > 0);
    assert.ok(scoreField('date', '2020-07-23', '2020-07-01') < 1);
    assert.equal(scoreField('date', '2020-07-23', '2019-01-01'), 0);
  });

  test('total: tolerates rounding but not real errors', () => {
    assert.equal(scoreField('total', 184.89, 184.89), 1);
    assert.equal(scoreField('total', 184.89, '184.89'), 1);
    assert.ok(scoreField('total', 184.89, 184.9) >= 0.9);
    assert.equal(scoreField('total', 184.89, 18.489), 0);
    assert.equal(scoreField('total', 184.89, null), 0);
  });

  test('currency: exact only', () => {
    assert.equal(scoreField('currency', 'GBP', 'GBP'), 1);
    assert.equal(scoreField('currency', 'GBP', 'USD'), 0);
    assert.equal(scoreField('currency', 'GBP', null), 0);
  });

  test('absent ground-truth field is not penalised', () => {
    assert.equal(scoreField('merchant', null, 'anything'), 1);
    assert.equal(scoreField('total', null, null), 1);
  });

  test('line_items: rewards matching descriptions and amounts', () => {
    const expected = [
      { description: 'Coffee', amount: 3.5 },
      { description: 'Bagel', amount: 2.25 },
    ];
    assert.equal(scoreField('line_items', expected, expected), 1);
    assert.equal(
      scoreField('line_items', expected, [{ description: 'Coffee', amount: 3.5 }]),
      0.5,
    );
    assert.equal(scoreField('line_items', expected, []), 0);
  });
});

describe('scoreReceipt', () => {
  const expected = {
    merchant: 'GARCIA SUPERMARKET',
    date: '2020-07-23',
    total: 184.89,
    currency: 'GBP',
    tax: 30.81,
    line_items: [],
  };

  test('perfect output scores 1.0 and is an exact match', () => {
    const r = scoreReceipt({
      raw: '',
      parsed: { ...expected },
      expected,
    });
    assert.equal(r.accuracy, 1);
    assert.equal(r.exactMatch, true);
  });

  test('headline accuracy ignores tax and line_items', () => {
    const r = scoreReceipt({
      raw: '',
      parsed: { ...expected, tax: 0, line_items: null },
      expected,
    });
    assert.equal(r.accuracy, 1, 'tax/line_items must not affect headline accuracy');
    assert.ok(r.accuracyAllFields < 1, 'all-field accuracy should reflect the misses');
  });

  test('unparseable output scores 0 and is flagged', () => {
    const r = scoreReceipt({ raw: 'sorry', parsed: null, expected });
    assert.equal(r.accuracy, 0);
    assert.equal(r.parsedOk, false);
  });
});

describe('summarize', () => {
  test('aggregates accuracy, latency percentiles, and parse strategies', () => {
    const mk = (acc, ms, strategy) => ({
      score: { accuracy: acc, accuracyAllFields: acc, exactMatch: acc === 1, parsedOk: true, parseStrategy: strategy, fieldScores: { merchant: acc, date: acc, total: acc, currency: acc, tax: acc, line_items: acc } },
      metrics: { totalMs: ms, tokensPerSecond: 10 },
    });
    const s = summarize([mk(1, 100, 'direct'), mk(0.5, 200, 'direct'), mk(0, 300, 'repaired')]);
    assert.ok(Math.abs(s.accuracy - 0.5) < 1e-9);
    assert.equal(s.latencyMs.min, 100);
    assert.equal(s.latencyMs.max, 300);
    assert.equal(s.parseStrategies.direct, 2);
    assert.equal(s.parseStrategies.repaired, 1);
    assert.ok(Math.abs(s.fieldAccuracy.merchant - 0.5) < 1e-9);
  });

  test('handles an empty result set without dividing by zero', () => {
    const s = summarize([]);
    assert.equal(s.accuracy, 0);
    assert.equal(s.latencyMs.p50, null);
  });
});
