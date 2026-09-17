/**
 * JSON Schema for the strict transaction contract, used to drive constrained
 * decoding (`@huggingface/transformers-structured-output`) during generation.
 *
 * This exists to fix a measured failure mode, not for tidiness. SmolVLM-256M
 * reads a receipt correctly and then wraps values in invented nested objects
 * ("total_price", "creditcardprice", "service_price") while emitting invalid
 * numbers like `+345150-243`. Those keys are CORD-dataset vocabulary bleeding
 * through from training, so prompt wording alone does not reliably suppress
 * them. Constraining the grammar removes the entire class of failure.
 *
 * Notes on the profile the constrained engine supports:
 *  - `anyOf: [{type:'number'}, {type:'null'}]` is used instead of the newer
 *    `type: ['number','null']` union syntax, which is not guaranteed.
 *  - String `pattern`/`format` are deliberately absent: the engine rejects them
 *    because they cannot be enforced incrementally.
 *  - Key order here is the order the model must emit, so the most important
 *    field (`total`) comes early where truncation is least likely.
 */
export const RECEIPT_JSON_SCHEMA = {
  type: 'object',
  properties: {
    merchant: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    date: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    total: { anyOf: [{ type: 'number' }, { type: 'null' }] },
    currency: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    tax: { anyOf: [{ type: 'number' }, { type: 'null' }] },
    line_items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          description: { anyOf: [{ type: 'string' }, { type: 'null' }] },
          amount: { anyOf: [{ type: 'number' }, { type: 'null' }] },
        },
        required: ['description', 'amount'],
        additionalProperties: false,
      },
    },
  },
  required: ['merchant', 'date', 'total', 'currency', 'tax', 'line_items'],
  additionalProperties: false,
};

/**
 * A reduced schema with no arrays. Some small models degrade badly when forced
 * through a variable-length array grammar, and for the personal-finance use case
 * the four scalar fields are what actually create a transaction.
 */
export const RECEIPT_JSON_SCHEMA_SCALARS_ONLY = {
  type: 'object',
  properties: {
    total: { anyOf: [{ type: 'number' }, { type: 'null' }] },
    currency: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    merchant: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    date: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    tax: { anyOf: [{ type: 'number' }, { type: 'null' }] },
  },
  required: ['total', 'currency', 'merchant', 'date', 'tax'],
  additionalProperties: false,
};

export const SCHEMAS = {
  full: RECEIPT_JSON_SCHEMA,
  scalars: RECEIPT_JSON_SCHEMA_SCALARS_ONLY,
};
