/**
 * Prompt variants for receipt extraction.
 *
 * Prompt wording is the single biggest lever on sub-1B VLM accuracy: these
 * models fail less often because they can't read the receipt than because they
 * wrap output in prose, invent fields, or truncate before the closing brace.
 *
 * Every variant therefore (a) states the exact key set, (b) demands JSON only,
 * (c) defines "null" for absent fields, and (d) puts the fields in a fixed order
 * so decoding stays on-rails. Variants are kept separate and benchmarked rather
 * than merged, so we can attribute accuracy to the prompt.
 *
 * `schemaVersion` identifies the output contract the grader expects.
 */

const FIELD_SPEC = `{
  "merchant": string or null,
  "date": string or null,
  "total": number or null,
  "currency": string or null,
  "tax": number or null,
  "line_items": array
}`;

export const PROMPT_V1 = `You are a receipt parser. Read the receipt image and output JSON.

Output ONLY a JSON object with exactly these keys:
${FIELD_SPEC}

Rules:
- "total" is the final amount paid, as a number with no currency symbol.
- "tax" is the tax or VAT amount, as a number. null if not shown.
- "currency" is the 3-letter code, e.g. "USD", "GBP", "EUR".
- "date" is the purchase date as YYYY-MM-DD.
- "merchant" is the store name.
- "line_items" is an array of {"description": string, "amount": number}. Empty array if not shown.
- Use null for anything you cannot read. Do not guess.
- No explanation, no markdown, no code fences. Just the JSON object.`;

export const PROMPT_V2_TERSE = `Extract the receipt as JSON. Keys: merchant, date (YYYY-MM-DD), total, currency (3-letter), tax, line_items (array of {description, amount}). Use null if absent. Reply with JSON only.`;

export const PROMPT_V3_FEWSHOT = `You convert receipt images into JSON for an expense tracker.

Example output:
{"merchant":"CORNER CAFE","date":"2024-03-12","total":18.75,"currency":"USD","tax":1.5,"line_items":[{"description":"Latte","amount":4.5},{"description":"Bagel","amount":3.25}]}

Now output the same JSON shape for the receipt image.
Keys: ${FIELD_SPEC}
Use null for values you cannot read. Reply with JSON only, no markdown.`;

export const PROMPT_V4_INVOICE = `Read this receipt. Output a single JSON object describing the purchase.

{
  "merchant": "name of the shop or restaurant",
  "date": "purchase date as YYYY-MM-DD",
  "total": "final total paid, number only",
  "currency": "ISO 4217 code such as USD or EUR",
  "tax": "tax amount, number only",
  "line_items": [{"description": "item name", "amount": "item price, number only"}]
}

If a value is not visible on the receipt, use null. Never invent values.
Answer with the JSON object only — no prose, no markdown fences.`;

/**
 * Puts `total` first. Autoregressive decoders commit to early tokens, and tiny
 * models that run out of budget mid-object lose whatever comes last — measured
 * on SmolVLM-256M, `total` and `currency` scored 0% while `date` (a later key in
 * v1) scored 37%, so key order is worth testing explicitly.
 */
export const PROMPT_V5_TOTAL_FIRST = `You are a receipt reader. Output one JSON object ONLY.

Start with "total". Keys in this exact order:
{
  "total": final amount paid as a number (no currency symbol),
  "currency": 3-letter code such as "USD" or "EUR",
  "merchant": store name,
  "date": purchase date as YYYY-MM-DD,
  "tax": tax or VAT amount as a number, or null,
  "line_items": [{"description": "item name", "amount": 0.00}]
}

Rules:
- Read the numbers from the receipt image. Do not invent values.
- Use null when a value is missing.
- Output must start with { and end with }.
- No prose, no markdown, no code fences.`;

/**
 * Mirrors the receipt's own vocabulary. Receipts print "TOTAL", "TAX"/"VAT",
 * "SUBTOTAL", so using the same words reduces the gap between the visual text
 * and the requested key for a model with limited instruction following.
 */
export const PROMPT_V6_RECEIPT_TERMS = `Look at the receipt and fill in this form. Reply with JSON only.

Ask yourself these questions and answer each one:
1. What is printed next to "TOTAL" or "AMOUNT DUE"? -> total
2. What currency symbol is printed ($, £, €)? -> currency code
3. What is the shop name at the top? -> merchant
4. What is the purchase date? -> date as YYYY-MM-DD
5. What is printed next to "TAX", "VAT" or "GST"? -> tax
6. What are the item lines and their prices? -> line_items

{"total": null, "currency": null, "merchant": null, "date": null, "tax": null, "line_items": []}

Replace the nulls with the real values. Keep the same keys and the same order.
JSON only — no explanation, no markdown.`;

export const PROMPTS = {
  v1: { id: 'v1', text: PROMPT_V1, description: 'Detailed rules, fixed key order, explicit null policy.' },
  v2: { id: 'v2', text: PROMPT_V2_TERSE, description: 'One-line terse instruction.' },
  v3: { id: 'v3', text: PROMPT_V3_FEWSHOT, description: 'Includes a worked example output.' },
  v4: { id: 'v4', text: PROMPT_V4_INVOICE, description: 'Field-by-field descriptions, prose-averse.' },
  v5: { id: 'v5', text: PROMPT_V5_TOTAL_FIRST, description: 'Total-first key order to survive truncation.' },
  v6: { id: 'v6', text: PROMPT_V6_RECEIPT_TERMS, description: "Uses the receipt's own vocabulary (TOTAL/VAT)." },
};

export const DEFAULT_PROMPT = 'v1';

export function getPrompt(id) {
  const p = PROMPTS[id];
  if (!p) throw new Error(`Unknown prompt variant: ${id}`);
  return p;
}
