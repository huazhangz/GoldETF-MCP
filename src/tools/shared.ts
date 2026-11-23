import { z } from "zod";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { FredClient, FredError } from "../fred/client.js";
import { applyTransform, type Point, type Transform } from "../stats/transforms.js";

/* ------------------------------------------------------------------ */
/* Shared input fragments. Descriptions are written for the model.     */
/* ------------------------------------------------------------------ */

export const seriesId = z
  .string()
  .trim()
  .min(1)
  .max(64)
  .transform((s) => s.toUpperCase())
  .describe('FRED series ID, e.g. "GVZCLS" (gold ETF volatility), "DGS10", "CPIAUCSL". Use fred_search_series if unknown.');

export const dateStr = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/, "Use YYYY-MM-DD.")
  .describe("Date in YYYY-MM-DD format.");

export const frequency = z
  .enum(["d", "w", "bw", "m", "q", "sa", "a"])
  .describe(
    "Aggregate to a LOWER frequency on FRED's side: d=daily, w=weekly, bw=biweekly, m=monthly, q=quarterly, sa=semiannual, a=annual. Omit to keep the native frequency. Cannot convert to a higher frequency.",
  );

export const aggregation = z
  .enum(["avg", "sum", "eop"])
  .describe("How to aggregate when `frequency` is set: avg (default), sum, or eop (end of period).");

export const transform = z
  .enum(["none", "log", "diff", "log_diff", "seasonal_diff", "log_diff_seasonal_diff"])
  .describe(
    "Transform applied before analysis. log_diff = first difference of the natural log (approx. % change); seasonal_diff uses `seasonal_period`.",
  );

/** Fields shared by every tool that analyzes a series. Either `series_id` or `values` is required. */
export const seriesSourceShape = {
  series_id: seriesId.optional(),
  values: z
    .array(z.number().finite())
    .min(10)
    .max(20_000)
    .optional()
    .describe("Inline numeric series (oldest first). Use instead of series_id for data not on FRED, e.g. model residuals."),
  start: dateStr.optional(),
  end: dateStr.optional(),
  frequency: frequency.optional(),
  aggregation: aggregation.optional(),
  transform: transform.optional(),
  seasonal_period: z
    .number()
    .int()
    .min(2)
    .max(366)
    .optional()
    .describe("Season length for seasonal_diff (12 for monthly, 4 for quarterly). Default 12."),
};

type SourceArgs = {
  series_id?: string;
  values?: number[];
  start?: string;
  end?: string;
  frequency?: z.infer<typeof frequency>;
  aggregation?: z.infer<typeof aggregation>;
  transform?: Transform;
  seasonal_period?: number;
};

export interface LoadedSeries {
  label: string;
  points: Point[];
  droppedMissing: number;
}

/** Resolve a series from FRED or inline values, drop missing values, apply the requested transform. */
export async function loadSeries(client: FredClient, args: SourceArgs, transformOverride?: Transform): Promise<LoadedSeries> {
  if (Boolean(args.series_id) === Boolean(args.values)) {
    throw new ToolInputError('Provide exactly one of "series_id" (a FRED series) or "values" (an inline numeric array).');
  }
  let raw: Point[];
  let droppedMissing = 0;
  let label: string;
  if (args.values) {
    raw = args.values.map((value, i) => ({ date: `t${i + 1}`, value }));
    label = "inline values";
  } else {
    const obs = await client.getObservations(args.series_id!, {
      start: args.start,
      end: args.end,
      frequency: args.frequency,
      aggregation: args.aggregation,
    });
    raw = [];
    for (const o of obs) {
      if (o.value === null || Number.isNaN(o.value)) droppedMissing++;
      else raw.push({ date: o.date, value: o.value });
    }
    label = args.series_id! + (args.frequency ? ` (${args.frequency}, ${args.aggregation ?? "avg"})` : "");
  }
  const t = transformOverride ?? args.transform ?? "none";
  const points = applyTransform(raw, t, args.seasonal_period ?? 12);
  if (points.length < 10) {
    throw new ToolInputError(
      `Only ${points.length} usable observations after transform "${t}". Widen the date range or use a higher frequency.`,
    );
  }
  return { label, points, droppedMissing };
}

/* ------------------------------------------------------------------ */
/* Results & errors                                                    */
/* ------------------------------------------------------------------ */

/** An error caused by the caller's arguments; the message tells the model how to fix the call. */
export class ToolInputError extends Error {}

/** Round to `sig` significant digits so outputs don't waste tokens on float noise. */
export function round(x: number, sig = 6): number {
  if (!Number.isFinite(x) || x === 0) return x;
  return Number(x.toPrecision(sig));
}

export function ok(payload: unknown): CallToolResult {
  return { content: [{ type: "text", text: JSON.stringify(payload, null, 2) }] };
}

const HINTS: Record<string, string> = {
  missing_api_key: "This tool needs FRED access; tools given inline `values` still work without a key.",
  invalid_api_key: "Ask the user to check FRED_API_KEY (32 lowercase alphanumeric characters).",
  not_found: "Call fred_search_series to find the correct series ID.",
  rate_limited: "Wait about a minute before retrying.",
  bad_request: "Check the arguments: dates must be YYYY-MM-DD and `frequency` can only aggregate to a lower frequency than the series' native one.",
};

/**
 * Wrap a tool handler so every failure becomes an `isError` result the model can read and act on,
 * instead of a protocol error the client may hide.
 */
export function safe<A>(fn: (args: A) => Promise<CallToolResult>) {
  return async (args: A): Promise<CallToolResult> => {
    try {
      return await fn(args);
    } catch (err) {
      let text: string;
      if (err instanceof FredError) {
        const hint = HINTS[err.kind];
        text = `FRED error (${err.kind}): ${err.message}${hint ? `\nHint: ${hint}` : ""}`;
      } else if (err instanceof ToolInputError) {
        text = `Invalid arguments: ${err.message}`;
      } else {
        text = `Error: ${err instanceof Error ? err.message : String(err)}`;
      }
      return { isError: true, content: [{ type: "text", text }] };
    }
  };
}
