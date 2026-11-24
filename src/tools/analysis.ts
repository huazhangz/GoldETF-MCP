import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { FredClient } from "../fred/client.js";
import { adfTest } from "../stats/adf.js";
import { kpssTest } from "../stats/kpss.js";
import { acf, ljungBox, pacf } from "../stats/autocorrelation.js";
import { archLm, jarqueBera } from "../stats/diagnostics.js";
import { stl } from "../stats/stl.js";
import type { Transform } from "../stats/transforms.js";
import { dataNote, loadSeries, ok, round, roundP, safe, seriesSourceShape, ToolInputError, transform } from "./shared.js";

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
        'to compare e.g. ["log", "log_diff"] and pick the transform for modeling. ' +
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
      let note: ReturnType<typeof dataNote> | undefined;
      for (const t of list) {
        const s = await loadSeries(client, args, t);
        note ??= dataNote(s);
        const x = s.points.map((p) => p.value);
        const adf = adfTest(x, { regression: args.regression, lags: args.adf_lags });
        const kpss = kpssTest(x, { regression: args.regression });
        results.push({
          transform: t,
          n: x.length,
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
      const { transform: _t, n: _n, ...rest } = note!;
      return ok({ ...rest, regression: args.regression, alpha: args.alpha, results });
    }),
  );

  server.registerTool(
    "ts_autocorrelation",
    {
      title: "Autocorrelation and ARCH diagnostics",
      description:
        "Linear and nonlinear dependence diagnostics: ACF and PACF up to `max_lag` with lags outside ±1.96/√n flagged, " +
        "the Ljung-Box white-noise test, Engle's ARCH-LM test and the ACF of squared values (volatility clustering), " +
        "and Jarque-Bera normality. Use it to choose ARIMA orders (PACF cutoff suggests AR p, ACF cutoff suggests MA q), " +
        "to decide whether a GARCH model is warranted (only if ARCH-LM rejects), or with inline `values` = model residuals " +
        "and `fit_df` = p+q to check residuals. Usually run on a stationary transform such as log_diff.",
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
        arch_lags: z.number().int().min(1).max(50).default(12).describe("Lags in the ARCH-LM test. Default 12."),
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
      const m = x.reduce((u, v) => u + v, 0) / x.length;
      const centered = x.map((v) => v - m);
      const sq = centered.map((v) => v * v);
      const aSq = acf(sq, args.max_lag).slice(1);
      const bound = 1.96 / Math.sqrt(x.length);
      const sig = (arr: number[]) => arr.flatMap((v, i) => (Math.abs(v) > bound ? [i + 1] : []));
      const lb = ljungBox(x, args.max_lag, args.fit_df);
      const lbLags = [...new Set([6, 12, 18, 24, args.max_lag])].filter((l) => l <= args.max_lag);
      const minP = lb.reduce((mm, r) => Math.min(mm, r.pValue), 1);
      const arch = archLm(centered, Math.min(args.arch_lags, Math.floor(x.length / 4)));
      const jb = jarqueBera(x);
      return ok({
        ...dataNote(s),
        confidence_bound: round(bound, 4),
        significant_acf_lags: sig(a),
        significant_pacf_lags: sig(p),
        acf: a.map((v) => round(v, 4)),
        pacf: p.map((v) => round(v, 4)),
        ljung_box: lb
          .filter((r) => lbLags.includes(r.lag))
          .map((r) => ({ lag: r.lag, statistic: round(r.statistic, 5), df: r.df, p_value: roundP(r.pValue) })),
        white_noise:
          lb.length === 0
            ? "not tested (fit_df >= max_lag)"
            : minP >= 0.05
              ? "consistent with white noise at 5% (no Ljung-Box lag rejects)"
              : `autocorrelation remains (smallest Ljung-Box p = ${round(minP, 3)} across lags 1-${args.max_lag})`,
        arch_effects: {
          arch_lm: { lags: arch.df, statistic: round(arch.statistic, 5), p_value: roundP(arch.pValue) },
          significant_squared_acf_lags: sig(aSq),
          conclusion:
            arch.pValue < 0.05
              ? "ARCH effects present (volatility clustering): a GARCH-type variance model is warranted."
              : "No ARCH effects detected: a GARCH model is not warranted at this frequency; constant variance is adequate.",
        },
        normality: {
          jarque_bera: round(jb.statistic, 5),
          p_value: roundP(jb.pValue),
          skewness: round(jb.skewness, 4),
          kurtosis: round(jb.kurtosis, 4),
          conclusion:
            jb.pValue < 0.05
              ? `Not normal (skewness ${jb.skewness.toFixed(2)}, kurtosis ${jb.kurtosis.toFixed(2)} vs 3): Gaussian intervals will misstate tail risk; consider a Student-t distribution.`
              : "Consistent with normality.",
        },
        note: "With many lags, ~5% of ACF/PACF values exceed the band by chance alone; isolated small spikes are weak evidence.",
      });
    }),
  );

  server.registerTool(
    "ts_decompose_stl",
    {
      title: "STL decomposition",
      description:
        "Decompose a series into trend + seasonal + remainder with STL (as R's stl()) and report trend strength F_T and " +
        "seasonal strength F_S (0-1; above ~0.6 is strong), the average seasonal effect for each position in the cycle, " +
        "and the dates of the largest remainder shocks. Use it to decide whether seasonal terms (SARIMA, seasonal " +
        "differencing) are worth modeling. Typically run on the log level, not on differences.",
      inputSchema: {
        ...seriesSourceShape,
        period: z.number().int().min(2).max(366).default(12).describe("Cycle length: 12 for monthly, 4 for quarterly, 52 for weekly. Default 12."),
        seasonal_window: z
          .union([z.literal("periodic"), z.number().int().min(3).max(999)])
          .default("periodic")
          .describe('Seasonal smoother span (odd integer >= 7), or "periodic" for a fixed seasonal pattern (R default for s.window = "periodic").'),
        robust: z.boolean().default(true).describe("Downweight outliers (crisis spikes). Default true."),
        top_shocks: z.number().int().min(0).max(30).default(8).describe("How many of the largest remainder values to list. Default 8."),
      },
      annotations: ANALYSIS,
    },
    safe(async (args) => {
      const s = await loadSeries(client, args);
      const x = s.points.map((p) => p.value);
      const r = stl(x, { period: args.period, seasonal: args.seasonal_window, robust: args.robust });
      const effects = Array.from({ length: args.period }, (_, k) => {
        const vals = r.seasonal.filter((_, i) => i % args.period === k);
        return vals.reduce((u, v) => u + v, 0) / vals.length;
      });
      // Label cycle positions by calendar month when the data are monthly and dated.
      const firstDate = s.points[0]!.date;
      const monthly = args.period === 12 && /^\d{4}-\d{2}-01$/.test(firstDate);
      const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const offset = monthly ? Number(firstDate.slice(5, 7)) - 1 : 0;
      const seasonal_effects = effects
        .map((e, k) => ({ position: monthly ? months[(k + offset) % 12]! : `${k + 1}`, effect: round(e, 4) }))
        .sort((a, b) => (monthly ? months.indexOf(a.position) - months.indexOf(b.position) : 0));
      const amp = Math.max(...effects) - Math.min(...effects);
      const remSd = Math.sqrt(r.remainder.reduce((u, v) => u + v * v, 0) / r.remainder.length);
      const shocks = r.remainder
        .map((v, i) => ({ date: s.points[i]!.date, remainder: round(v, 4), robustness_weight: round(r.weights[i]!, 3) }))
        .sort((a, b) => Math.abs(b.remainder) - Math.abs(a.remainder))
        .slice(0, args.top_shocks);
      return ok({
        ...dataNote(s),
        trend_strength: round(r.trendStrength, 4),
        seasonal_strength: round(r.seasonalStrength, 4),
        seasonal_peak_to_trough: round(amp, 4),
        remainder_rms: round(remSd, 4),
        interpretation:
          (r.seasonalStrength < 0.3
            ? `Seasonality is negligible (F_S = ${r.seasonalStrength.toFixed(3)}; peak-to-trough ${amp.toFixed(3)} vs remainder RMS ${remSd.toFixed(3)}): seasonal terms are not worth modeling. `
            : r.seasonalStrength < 0.6
              ? `Seasonality is moderate (F_S = ${r.seasonalStrength.toFixed(3)}). `
              : `Seasonality is strong (F_S = ${r.seasonalStrength.toFixed(3)}): consider seasonal differencing or SARIMA terms. `) +
          (r.trendStrength > 0.6 ? `Trend/cycle is strong (F_T = ${r.trendStrength.toFixed(3)}).` : `Trend is weak (F_T = ${r.trendStrength.toFixed(3)}).`),
        seasonal_effects,
        largest_shocks: shocks,
        stl_parameters: r.params,
      });
    }),
  );
}
