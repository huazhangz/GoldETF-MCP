/**
 * Measured spend of an eval flow, from the `model` and `usage` recorded on every row (results.jsonl and
 * errors.jsonl, which also holds billed-but-failed attempts). Prices: first-party USD per million tokens.
 *
 *   npx tsx evals/cost.ts .claude/hillclimb/tool-use
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const PRICES: Record<string, { in: number; out: number }> = {
  "claude-opus-5-5": { in: 4, out: 20 },
  "claude-sonnet-5-5": { in: 2, out: 10 },
  "claude-haiku-5-5": { in: 0.1, out: 0.5 },
  "claude-fable-5-2": { in: 8, out: 40 },
};
type Usage = { input_tokens: number; output_tokens: number; cache_read_input_tokens?: number; cache_creation_input_tokens?: number };

export function rowCost(model: string | undefined, u: Usage | undefined): number {
  if (!model || !u) return 0;
  const p = PRICES[model];
  if (!p) throw new Error(`no price for model ${model}`);
  return (
    (u.input_tokens * p.in + (u.cache_creation_input_tokens ?? 0) * p.in * 1.25 + (u.cache_read_input_tokens ?? 0) * p.in * 0.1 + u.output_tokens * p.out) /
    1e6
  );
}

const flow = process.argv[2] ?? ".claude/hillclimb/tool-use";
let total = 0;
for (const v of readdirSync(flow).filter((d) => /^(baseline|v\d+)$/.test(d)).sort()) {
  const rows = ["results.jsonl", "errors.jsonl"].flatMap((f) => {
    const p = join(flow, v, f);
    return existsSync(p) ? readFileSync(p, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
  });
  const costs = rows.map((r) => rowCost(r.model, r.usage));
  const sum = costs.reduce((a, b) => a + b, 0);
  total += sum;
  const sorted = [...costs].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)] ?? 0;
  console.log(
    `${v}: ${rows.length} attempts, $${sum.toFixed(3)} (per attempt min/median/max $${(sorted[0] ?? 0).toFixed(3)}/$${med.toFixed(3)}/$${(sorted.at(-1) ?? 0).toFixed(3)})`,
  );
}
console.log(`total: $${total.toFixed(3)}`);
