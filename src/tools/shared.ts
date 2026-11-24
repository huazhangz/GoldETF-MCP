import { z } from "zod";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { FredClient, FredError } from "../fred/client.js";
import { aggregate as aggregatePoints } from "../stats/aggregate.js";
import { applyTransform, type Point, type Transform } from "../stats/transforms.js";

/* ------------------------------------------------------------------ */
/* Shared input fragments. Descriptions are written for the model.     */
/* ------------------------------------------------------------------ */

// Factories, not shared instances: reusing one zod instance twice in a tool makes zod-to-json-schema emit
// "$ref" pointers, which some MCP clients do not resolve. Descriptions are terse because every tool
// definition costs context in every client session.
export const seriesId = () =>
  z
    .string()
    .trim()
    .min(1)
    .max(64)
    .transform((s) => s.toUpperCase())
    .describe('FRED series ID, e.g. "CPIAUCSL" (find IDs with fred_search_series).');

export const dateStr = (what = "Date") => z.string().date("Use YYYY-MM-DD.").describe(what);

export const frequency = () =>
  z.enum(["w", "m", "q", "sa", "a"]).describe("Aggregate to a lower frequency first (w=weekly, m, q, sa, a). Omit for native.");

export const aggregation = () => z.enum(["avg", "sum", "eop"]).describe("With frequency: avg (default), sum, or eop (end of period).");

export const TRANSFORMS = ["none", "log", "diff", "log_diff", "seasonal_diff", "log_seasonal_diff", "log_diff_seasonal_diff"] as const;
export const transform = () =>
  z.enum(TRANSFORMS).describe("log_diff = change of log (approx. % change); seasonal_* difference at lag seasonal_period.");

/**
 * Fields shared by the analysis tools. Either `series_id` or `values` is required. Rarely needed options
 * (aggregation method, partial periods, seasonal period) live only on fred_get_observations; analysis
 * tools use averages, drop partial periods and infer the seasonal period from the frequency.
 */
export const seriesSourceShape = () => ({
  series_id: seriesId().optional(),
  values: z
    .array(z.number().finite())
    .min(10)
    .max(20_000)
    .optional()
    .describe("Inline series, oldest first, instead of series_id (e.g. model residuals)."),
  start: dateStr("Start date").optional(),
  end: dateStr("End date").optional(),
  frequency: frequency().optional(),
  as_of: dateStr("ALFRED vintage: data as published on this date, before revisions").optional(),
  transform: transform().optional(),
});

/** Season length implied by the target frequency, or by the native spacing of the data. */
export function inferSeasonalPeriod(freq: string | undefined, points: Point[]): number {
  const byFreq: Record<string, number> = { w: 52, m: 12, q: 4, sa: 2, a: 1 };
  if (freq) return byFreq[freq]!;
  const day = 86_400_000;
  const gaps = points.slice(1, 200).map((p, i) => (Date.parse(p.date) - Date.parse(points[i]!.date)) / day);
  const g = gaps.filter(Number.isFinite).sort((a, b) => a - b)[Math.floor(gaps.length / 2)];
  if (g === undefined) return 12;
  if (g >= 6 && g <= 8) return 52;
  if (g >= 80 && g <= 100) return 4;
  if (g >= 170 && g <= 200) return 2;
  if (g >= 350) return 1;
  return 12;
}

export type SourceArgs = {
  series_id?: string;
  values?: number[];
  start?: string;
  end?: string;
  frequency?: z.infer<ReturnType<typeof frequency>>;
  aggregation?: z.infer<ReturnType<typeof aggregation>>;
  keep_incomplete_periods?: boolean;
  as_of?: string;
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
    const obs = await client.getObservations(args.series_id!, { start: args.start, end: args.end, asOf: args.as_of });
    raw = [];
    for (const o of obs) {
      if (o.value === null || Number.isNaN(o.value)) droppedMissing++;
      else raw.push({ date: o.date, value: o.value });
    }
    if (raw.length === 0) throw new ToolInputError(`No observations for ${args.series_id} in the requested date range.`);
    label = args.series_id! + (args.as_of ? ` [vintage ${args.as_of}]` : "");
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
  const period = args.seasonal_period ?? inferSeasonalPeriod(args.frequency, raw);
  if (t.includes("seasonal") && period < 2) {
    throw new ToolInputError(`Seasonal transforms need a seasonal period of at least 2; ${args.frequency ?? "this"} data has none.`);
  }
  const points = applyTransform(raw, t, period);
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

/** Combined ADF (H0 unit root) + KPSS (H0 stationary) verdict. */
export function stationarityVerdict(adfP: number, kpssP: number, alpha: number): { conclusion: string; explanation: string } {
  const adfRejects = adfP < alpha;
  const kpssRejects = kpssP < alpha;
  if (adfRejects && !kpssRejects) {
    return { conclusion: "stationary", explanation: "ADF rejects a unit root and KPSS does not reject stationarity." };
  }
  if (!adfRejects && kpssRejects) {
    return {
      conclusion: "non_stationary",
      explanation: 'ADF cannot reject a unit root and KPSS rejects stationarity. Difference the series (try transform "diff" or "log_diff").',
    };
  }
  if (adfRejects && kpssRejects) {
    return {
      conclusion: "conflicting",
      explanation:
        "Both tests reject. Typical of persistent but mean-reverting series (near unit root / long memory) or structural breaks. " +
        'Do not difference automatically: compare a stationary AR model on levels (d=0) with a d=1 model using ts_fit_arima and ts_backtest, or retry with regression "ct".',
    };
  }
  return { conclusion: "inconclusive", explanation: "Neither test rejects; the sample may be too short for the tests to have power." };
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
  missing_api_key: "Ask the user to configure FRED_API_KEY. Tools given inline `values` still work without a key.",
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
