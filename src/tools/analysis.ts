import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { FredClient } from "../fred/client.js";
import { adfTest } from "../stats/adf.js";
import { kpssTest } from "../stats/kpss.js";
import { acf, ljungBox, pacf } from "../stats/autocorrelation.js";
import type { Transform } from "../stats/transforms.js";
import { loadSeries, ok, round, safe, seriesSourceShape, ToolInputError, transform } from "./shared.js";

const ANALYSIS = { readOnlyHint: true, openWorldHint: true, idempotentHint: true } as const;

function verdict(adfP: number, kpssP: number, alpha: number): { conclusion: string; explanation: string } {
  const adfRejects = adfP < alpha;
  const kpssRejects = kpssP < alpha;
  if (adfRejects && !kpssRejects) {
    return { conclusion: "stationary", explanation: "ADF rejects a unit root and KPSS does not reject stationarity." };
  }
  if (!adfRejects && kpssRejects) {
    return {
      conclusion: "non_stationary",
      explanation: "ADF cannot reject a unit root and KPSS rejects stationarity. Difference the series (try transform \"diff\" or \"log_diff\").",
    };
  }
  if (adfRejects && kpssRejects) {
    return {
      conclusion: "conflicting",
      explanation:
        "Both tests reject. Common causes: a deterministic trend (retry with regression \"ct\"), structural breaks, or long memory.",
    };
  }
  return {
    conclusion: "inconclusive",
    explanation: "Neither test rejects; the sample may be too short for the tests to have power.",
  };
}

export function registerAnalysisTools(server: McpServer, client: FredClient) {
  server.registerTool(
    "ts_test_stationarity",
    {
      title: "Test stationarity (ADF + KPSS)",
      description:
        "Run the Augmented Dickey-Fuller test (H0: unit root) and the KPSS test (H0: stationary) together and return a " +
        "combined verdict: stationary, non_stationary, conflicting, or inconclusive. Pass several `transforms` in one call " +
        'to compare e.g. ["none", "log", "log_diff"] and find the transform that yields a stationary series. ' +
        "Works on a FRED series_id (fetched for you) or inline `values`.",
      inputSchema: {
        ...seriesSourceShape,
        transforms: z
          .array(transform)
          .min(1)
          .max(6)
          .optional()
          .describe('Transforms to compare side by side. Overrides `transform`. Example: ["log", "log_diff"].'),
        regression: z
          .enum(["c", "ct"])
          .default("c")
          .describe('Deterministic terms: "c" = constant (level stationarity), "ct" = constant + linear trend.'),
        adf_lags: z
          .number()
          .int()
          .min(0)
          .max(60)
          .optional()
          .describe("Fixed number of ADF lagged differences. Omit to select by AIC (recommended)."),
        alpha: z.number().gt(0).lt(0.5).default(0.05).describe("Significance level for the verdict. Default 0.05."),
      },
      annotations: ANALYSIS,
    },
    safe(async (args) => {
      const list: Transform[] = args.transforms ?? [args.transform ?? "none"];
      const results = [];
      let label = "";
      for (const t of list) {
        const s = await loadSeries(client, args, t);
        label = s.label;
        const x = s.points.map((p) => p.value);
        const adf = adfTest(x, { regression: args.regression, lags: args.adf_lags });
        const kpss = kpssTest(x, { regression: args.regression });
        results.push({
          transform: t,
          n: x.length,
          range: `${s.points[0]!.date} to ${s.points.at(-1)!.date}`,
          adf: {
            statistic: round(adf.statistic, 5),
            p_value: round(adf.pValue, 4),
            lags: adf.usedLag,
            lag_selection: adf.lagSelection,
            critical_values: Object.fromEntries(Object.entries(adf.criticalValues).map(([k, v]) => [k, round(v, 4)])),
          },
          kpss: {
            statistic: round(kpss.statistic, 5),
            p_value: kpss.pValue,
            ...(kpss.pValueBound !== "exact"
              ? { p_value_note: kpss.pValueBound === "lower" ? "true p-value < 0.01" : "true p-value > 0.10" }
              : {}),
            lags: kpss.lags,
          },
          ...verdict(adf.pValue, kpss.pValue, args.alpha),
        });
      }
      return ok({ series: label, regression: args.regression, alpha: args.alpha, results });
    }),
  );

  server.registerTool(
    "ts_autocorrelation",
    {
      title: "ACF, PACF and Ljung-Box",
      description:
        "Compute the autocorrelation (ACF) and partial autocorrelation (PACF) up to `max_lag`, flag lags outside the " +
        "±1.96/√n band, and run the Ljung-Box white-noise test. Use it to choose ARIMA orders (PACF cutoff suggests AR p, " +
        "ACF cutoff suggests MA q) or, with inline `values` = model residuals and `fit_df` = p+q, to check residuals " +
        "are white noise. Usually run on a stationary transform such as log_diff.",
      inputSchema: {
        ...seriesSourceShape,
        max_lag: z.number().int().min(1).max(200).default(24).describe("Largest lag to compute. Default 24."),
        fit_df: z
          .number()
          .int()
          .min(0)
          .max(50)
          .default(0)
          .describe("Degrees of freedom used by a fitted ARMA model (p+q) when testing its residuals. Default 0."),
      },
      annotations: ANALYSIS,
    },
    safe(async (args) => {
      const s = await loadSeries(client, args);
      const x = s.points.map((p) => p.value);
      if (args.max_lag >= x.length / 2) {
        throw new ToolInputError(`max_lag (${args.max_lag}) should be below n/2 = ${Math.floor(x.length / 2)}.`);
      }
      const a = acf(x, args.max_lag).slice(1);
      const p = pacf(x, args.max_lag);
      const bound = 1.96 / Math.sqrt(x.length);
      const sig = (arr: number[]) => arr.flatMap((v, i) => (Math.abs(v) > bound ? [i + 1] : []));
      const lb = ljungBox(x, args.max_lag, args.fit_df);
      // Report a few standard lags rather than all of them.
      const lbLags = [...new Set([6, 12, 18, 24, args.max_lag])].filter((l) => l <= args.max_lag);
      const lbShown = lb.filter((r) => lbLags.includes(r.lag));
      const minP = lb.reduce((m, r) => Math.min(m, r.pValue), 1);
      return ok({
        series: s.label,
        transform: args.transform ?? "none",
        n: x.length,
        confidence_bound: round(bound, 4),
        significant_acf_lags: sig(a),
        significant_pacf_lags: sig(p),
        acf: a.map((v) => round(v, 4)),
        pacf: p.map((v) => round(v, 4)),
        ljung_box: lbShown.map((r) => ({ lag: r.lag, statistic: round(r.statistic, 5), df: r.df, p_value: round(r.pValue, 4) })),
        white_noise:
          lb.length === 0
            ? "not tested (fit_df >= max_lag)"
            : minP >= 0.05
              ? "consistent with white noise at 5% (no Ljung-Box lag rejects)"
              : "autocorrelation remains (some Ljung-Box lag has p < 0.05)",
        note: "With many lags, ~5% of ACF/PACF values exceed the band by chance alone; isolated small spikes are weak evidence.",
      });
    }),
  );
}
