import { z } from "zod";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { FredClient, FredError } from "../fred/client.js";
import { aggregate as aggregatePoints } from "../stats/aggregate.js";
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
  .enum(["w", "m", "q", "sa", "a"])
  .describe(
    "Aggregate to a LOWER frequency before analysis: w=weekly (ending Friday), m=monthly, q=quarterly, sa=semiannual, a=annual. Omit to keep the native frequency.",
  );

export const aggregation = z
  .enum(["avg", "sum", "eop"])
  .describe("How to aggregate when `frequency` is set: avg (default), sum, or eop (end of period).");

export const transform = z
  .enum(["none", "log", "diff", "log_diff", "seasonal_diff", "log_seasonal_diff", "log_diff_seasonal_diff"])
  .describe(
    "Transform applied before analysis. log_diff = first difference of the natural log (approx. % change); the seasonal_* variants difference at lag `seasonal_period`.",
  );

/** Fields shared by every tool that reads a series. Either `series_id` or `values` is required. */
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
  keep_incomplete_periods: z
    .boolean()
    .optional()
    .describe(
      "When aggregating, keep a partially observed first/last period (e.g. the current month). Default false: partial periods are dropped because their averages are not comparable.",
    ),
  transform: transform.optional(),
  seasonal_period: z
    .number()
    .int()
    .min(2)
    .max(366)
    .optional()
    .describe("Season length for seasonal_diff (12 for monthly, 4 for quarterly). Default 12."),
};

export type SourceArgs = {
  series_id?: string;
  values?: number[];
  start?: string;
  end?: string;
  frequency?: z.infer<typeof frequency>;
  aggregation?: z.infer<typeof aggregation>;
  keep_incomplete_periods?: boolean;
  transform?: Transform;
  seasonal_period?: number;
};

export interface LoadedSeries {
  label: string;
  /** Series after aggregation and transform. */
  points: Point[];
  /** Series after aggregation, before the transform (original units). */
  levels: Point[];
  transform: Transform;
  droppedMissing: number;
  droppedIncomplete: string[];
}

/** Resolve a series from FRED or inline values, drop missing values, aggregate, and apply the transform. */
export async function loadSeries(
  client: FredClient,
  args: SourceArgs,
  transformOverride?: Transform,
  minPoints = 10,
): Promise<LoadedSeries> {
  if (Boolean(args.series_id) === Boolean(args.values)) {
    throw new ToolInputError('Provide exactly one of "series_id" (a FRED series) or "values" (an inline numeric array).');
  }
  let raw: Point[];
  let droppedMissing = 0;
  let droppedIncomplete: string[] = [];
  let label: string;
  if (args.values) {
    if (args.frequency) throw new ToolInputError('"frequency" only applies to FRED series; aggregate inline values yourself.');
    raw = args.values.map((value, i) => ({ date: `t${i + 1}`, value }));
    label = "inline values";
  } else {
    const obs = await client.getObservations(args.series_id!, { start: args.start, end: args.end });
    raw = [];
    for (const o of obs) {
      if (o.value === null || Number.isNaN(o.value)) droppedMissing++;
      else raw.push({ date: o.date, value: o.value });
    }
    if (raw.length === 0) throw new ToolInputError(`No observations for ${args.series_id} in the requested date range.`);
    label = args.series_id!;
    if (args.frequency) {
      const agg = aggregatePoints(raw, args.frequency, args.aggregation ?? "avg", {
        dropIncomplete: !(args.keep_incomplete_periods ?? false),
      });
      raw = agg.points;
      droppedIncomplete = agg.droppedIncomplete;
      label += ` (${args.frequency}, ${args.aggregation ?? "avg"})`;
    }
  }
  const t = transformOverride ?? args.transform ?? "none";
  const points = applyTransform(raw, t, args.seasonal_period ?? 12);
  if (points.length < minPoints) {
    throw new ToolInputError(
      `Only ${points.length} usable observations after transform "${t}" (need ${minPoints}). Widen the date range or use a higher frequency.`,
    );
  }
  return { label, points, levels: raw, transform: t, droppedMissing, droppedIncomplete };
}

/** Compact description of how the data were prepared, included in every analysis result. */
export function dataNote(s: LoadedSeries) {
  return {
    series: s.label,
    transform: s.transform,
    n: s.points.length,
    range: `${s.points[0]!.date} to ${s.points.at(-1)!.date}`,
    ...(s.droppedMissing ? { missing_dropped: s.droppedMissing } : {}),
    ...(s.droppedIncomplete.length ? { incomplete_periods_dropped: s.droppedIncomplete } : {}),
  };
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

/** Round p-values: 4 significant digits, and report tiny values as a bound. */
export function roundP(p: number): number | string {
  return p < 1e-4 ? "<1e-4" : round(p, 3);
}

export function ok(payload: unknown): CallToolResult {
  return { content: [{ type: "text", text: JSON.stringify(payload, null, 2) }] };
}

const HINTS: Record<string, string> = {
  missing_api_key: "Observation-based tools (fred_get_observations and all ts_* tools) still work without a key.",
  invalid_api_key: "Ask the user to check FRED_API_KEY (32 lowercase alphanumeric characters).",
  not_found: "Call fred_search_series to find the correct series ID.",
  rate_limited: "Wait about a minute before retrying.",
  bad_request: "Check the arguments: dates must be YYYY-MM-DD.",
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
