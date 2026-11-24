import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { FredClient } from "../fred/client.js";
import { adfTest } from "../stats/adf.js";
import { kpssTest } from "../stats/kpss.js";
import {
  correlationCI,
  correlationPValue,
  crossCorrelation,
  effectiveSampleSize,
  grangerTest,
  hacRegression,
  pearson,
  spearman,
} from "../stats/relationship.js";
import type { Transform } from "../stats/transforms.js";
import {
  dateStr,
  frequency,
  loadSeries,
  ok,
  round,
  roundP,
  safe,
  seriesId,
  seriesSourceShape,
  stationarityVerdict,
  ToolInputError,
  transform,
} from "./shared.js";

const ANALYSIS = { readOnlyHint: true, openWorldHint: true, idempotentHint: true } as const;
const inline = () => z.array(z.number().finite()).min(30).max(20_000).optional();

export function registerRelationshipTools(server: McpServer, client: FredClient) {
  server.registerTool(
    "ts_compare_series",
    {
      title: "Compare two series (correlation, lead/lag, Granger)",
      description:
        "Two series aligned by date: correlation with autocorrelation-adjusted p-value, lead/lag cross-correlation, Granger causality both ways, OLS slope with HAC errors. Warns when levels are non-stationary (likely spurious): compare changes (log_diff/diff).",
      inputSchema: {
        series_x: seriesId().optional().describe("FRED series ID of x (the candidate leader / regressor)."),
        series_y: seriesId().optional().describe("FRED series ID of y (the response)."),
        values_x: inline().describe("Inline x values (oldest first), instead of series_x. Must be the same length as values_y."),
        values_y: inline().describe("Inline y values (oldest first), instead of series_y."),
        start: dateStr().optional(),
        end: dateStr().optional(),
        frequency: frequency().optional(),
        as_of: seriesSourceShape().as_of,
        transform: transform().default("none").describe("Transform applied to both series unless transform_x / transform_y is set."),
        transform_x: transform().optional().describe("Transform for x only (e.g. log_diff for an index while y uses diff)."),
        transform_y: transform().optional().describe("Transform for y only."),
        max_lag: z.number().int().min(1).max(36).default(12).describe("Largest lead/lag for the cross-correlation. Default 12."),
        granger_lags: z.number().int().min(1).max(24).default(4).describe("Lags in the Granger causality regressions. Default 4."),
      },
      annotations: ANALYSIS,
    },
    safe(async (args) => {
      const fromFred = Boolean(args.series_x || args.series_y);
      const inlineData = Boolean(args.values_x || args.values_y);
      if (fromFred === inlineData) {
        throw new ToolInputError('Provide either series_x and series_y (FRED IDs) or values_x and values_y (inline arrays), not a mix.');
      }
      const tx: Transform = args.transform_x ?? args.transform;
      const ty: Transform = args.transform_y ?? args.transform;
      const common = {
        start: args.start,
        end: args.end,
        frequency: args.frequency,
        as_of: args.as_of,
      };
      let x: number[];
      let y: number[];
      let range: string;
      let labels: [string, string];
      if (fromFred) {
        if (!args.series_x || !args.series_y) throw new ToolInputError("Both series_x and series_y are required.");
        const [sx, sy] = await Promise.all([
          loadSeries(client, { ...common, series_id: args.series_x }, tx),
          loadSeries(client, { ...common, series_id: args.series_y }, ty),
        ]);
        const ym = new Map(sy.points.map((p) => [p.date, p.value]));
        const pairs = sx.points.filter((p) => ym.has(p.date)).map((p) => [p.date, p.value, ym.get(p.date)!] as const);
        if (pairs.length < 30) {
          throw new ToolInputError(
            `Only ${pairs.length} dates in common after aggregation and transforms. Use the same frequency for both (e.g. frequency "m") and widen the date range.`,
          );
        }
        x = pairs.map((p) => p[1]);
        y = pairs.map((p) => p[2]);
        range = `${pairs[0]![0]} to ${pairs.at(-1)![0]}`;
        labels = [sx.label, sy.label];
      } else {
        if (!args.values_x || !args.values_y || args.values_x.length !== args.values_y.length) {
          throw new ToolInputError("values_x and values_y must both be given and have the same length.");
        }
        const sx = await loadSeries(client, { values: args.values_x }, tx);
        const sy = await loadSeries(client, { values: args.values_y }, ty);
        const n = Math.min(sx.points.length, sy.points.length);
        x = sx.points.slice(-n).map((p) => p.value);
        y = sy.points.slice(-n).map((p) => p.value);
        range = `${n} aligned observations`;
        labels = ["inline x", "inline y"];
      }
      const n = x.length;
      if (args.max_lag >= n / 3) throw new ToolInputError(`max_lag (${args.max_lag}) should be below n/3 = ${Math.floor(n / 3)}.`);

      // Auxiliary check: on degenerate inputs (e.g. exactly periodic data) the ADF regression is singular,
      // which should not fail the whole comparison.
      const st = (v: number[]) => {
        try {
          return stationarityVerdict(adfTest(v).pValue, kpssTest(v).pValue, 0.05).conclusion;
        } catch {
          return "undetermined";
        }
      };
      const stationarity = { x: st(x), y: st(y) };
      const r = pearson(x, y);
      const rs = spearman(x, y);
      const nEff = effectiveSampleSize(x, y);
      const pNaive = correlationPValue(r, n);
      const pAdj = correlationPValue(r, nEff);
      const reg = hacRegression(y, x);
      const ccf = crossCorrelation(x, y, args.max_lag);
      const band = 1.96 / Math.sqrt(n);
      const strongest = ccf.reduce((a, b) => (Math.abs(b.r) > Math.abs(a.r) ? b : a));
      const g = Math.min(args.granger_lags, Math.floor((n - 10) / 3));
      let granger: Record<string, unknown>;
      try {
        const xToY = grangerTest(y, x, g);
        const yToX = grangerTest(x, y, g);
        const gv = (res: typeof xToY, from: string, to: string) =>
          res.pValue < 0.05 ? `${from} helps predict ${to} (p ${roundP(res.pValue)})` : `${from} does not help predict ${to} (p ${roundP(res.pValue)})`;
        granger = {
          lags: g,
          x_to_y: { F: round(xToY.F, 5), p_value: roundP(xToY.pValue) },
          y_to_x: { F: round(yToX.F, 5), p_value: roundP(yToX.pValue) },
          conclusion: `${gv(xToY, "x", "y")}; ${gv(yToX, "y", "x")}. Granger causality means predictive precedence, not causation.`,
        };
      } catch {
        granger = { lags: g, error: "Granger regressions are singular: x and y are (nearly) identical or exactly collinear." };
      }

      const warnings: string[] = [];
      const persistent = (["x", "y"] as const).filter((k) => stationarity[k] === "non_stationary" || stationarity[k] === "conflicting");
      if (persistent.length) {
        warnings.push(
          `${persistent.map((k) => (k === "x" ? labels[0] : labels[1])).join(" and ")} ${persistent.length > 1 ? "are" : "is"} not clearly stationary (${persistent
            .map((k) => stationarity[k])
            .join(", ")}): correlations and regressions between persistent levels are often spurious (Granger & Newbold 1974). ` +
            'Compare changes instead (transform "diff" or "log_diff"). Granger tests also assume stationary series.',
        );
      }
      if (nEff < n / 2) {
        warnings.push(
          `Both series are autocorrelated, so the effective sample size is about ${Math.round(nEff)} of ${n}; use p_value_adjusted, not p_value_naive.`,
        );
      }
      if (Math.abs(r - rs) > 0.2) {
        warnings.push(`Pearson (${r.toFixed(2)}) and Spearman (${rs.toFixed(2)}) differ: the linear correlation is driven by outliers or nonlinearity.`);
      }

      const sig = pAdj < 0.05;
      return ok({
        series_x: labels[0],
        series_y: labels[1],
        transform_x: tx,
        transform_y: ty,
        n,
        range,
        stationarity,
        correlation: {
          pearson: round(r, 4),
          spearman: round(rs, 4),
          n_effective: round(nEff, 4),
          p_value_naive: roundP(pNaive),
          p_value_adjusted: roundP(pAdj),
          ci95_adjusted: correlationCI(r, nEff).map((v) => round(v, 3)),
        },
        regression_y_on_x: {
          beta: round(reg.beta, 5),
          se_hac: round(reg.seBeta, 4),
          p_value: roundP(reg.pValue),
          r2: round(reg.r2, 4),
          hac_lags: reg.lags,
        },
        lead_lag: {
          convention: "lag k > 0 means x leads y by k periods: corr(x_t, y_{t+k})",
          strongest: { lag: strongest.lag, r: round(strongest.r, 4) },
          significant_lags: ccf.filter((c) => Math.abs(c.r) > band).map((c) => c.lag),
          band: round(band, 4),
          ccf: ccf.map((c) => [c.lag, round(c.r, 3)]),
        },
        granger,
        interpretation:
          `Correlation ${r.toFixed(3)} (adjusted p ${roundP(pAdj)}, effective n ≈ ${Math.round(nEff)}): ` +
          (sig ? "statistically significant at 5%." : "not statistically significant at 5%.") +
          (persistent.length ? " Treat with caution: see warnings (likely spurious)." : ""),
        warnings,
      });
    }),
  );
}
