import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { FredClient } from "../fred/client.js";
import { fitArima, forecastArima, type ArimaFit } from "../stats/arima.js";
import { ljungBox } from "../stats/autocorrelation.js";
import { backtest, modelLabel, type ModelSpec } from "../stats/backtest.js";
import { archLm, jarqueBera } from "../stats/diagnostics.js";
import { chiSquareSf } from "../stats/distributions.js";
import { ols } from "../stats/ols.js";
import { fitVolModel, forecastVol, intervalMultiplier, type VolFit, type VolSpec } from "../stats/volatility.js";
import { dataNote, loadSeries, ok, round, roundP, safe, seriesSourceShape, ToolInputError, type LoadedSeries } from "./shared.js";

const ANALYSIS = { readOnlyHint: true, openWorldHint: true, idempotentHint: true } as const;

/** Modeling tools take the level (optionally logged); differencing is part of the model (d). */
const levelTransform = z
  .enum(["none", "log"])
  .default("none")
  .describe('Use "log" for positive, right-skewed series such as volatility indices or prices. Forecasts are back-transformed to levels.');

const { transform: _omit, ...sourceShape } = seriesSourceShape;

const order = z
  .array(z.number().int().min(0).max(5))
  .length(3)
  .describe("[p, d, q] with p, q in 0..5 and d in 0..2, e.g. [1, 0, 0] for AR(1) on levels or [0, 1, 0] for a random walk.");

/* ------------------------------ helpers ------------------------------ */

function futureDates(s: LoadedSeries, freq: string | undefined, h: number): string[] {
  const last = s.points.at(-1)!.date;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(last);
  if (!m || !freq) return Array.from({ length: h }, (_, i) => `T+${i + 1}`);
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  const months = { m: 1, q: 3, sa: 6, a: 12 }[freq as "m" | "q" | "sa" | "a"];
  return Array.from({ length: h }, (_, i) => {
    const x = months
      ? new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + months * (i + 1), 1))
      : new Date(d.getTime() + 7 * 86_400_000 * (i + 1));
    return x.toISOString().slice(0, 10);
  });
}

function forecastTable(
  dates: string[],
  mean: number[],
  sd: number[],
  nu: number,
  logScale: boolean,
): Record<string, unknown>[] {
  const z80 = intervalMultiplier(0.8, nu);
  const z95 = intervalMultiplier(0.95, nu);
  return mean.map((mu, i) => {
    const s = sd[i]!;
    const row: Record<string, unknown> = {
      date: dates[i],
      mean: round(mu, 5),
      sd: round(s, 4),
      lo80: round(mu - z80 * s, 5),
      hi80: round(mu + z80 * s, 5),
      lo95: round(mu - z95 * s, 5),
      hi95: round(mu + z95 * s, 5),
    };
    if (logScale) {
      row.level = {
        median: round(Math.exp(mu), 5),
        ...(Number.isFinite(nu) ? {} : { mean: round(Math.exp(mu + (s * s) / 2), 5) }),
        lo80: round(Math.exp(mu - z80 * s), 5),
        hi80: round(Math.exp(mu + z80 * s), 5),
        lo95: round(Math.exp(mu - z95 * s), 5),
        hi95: round(Math.exp(mu + z95 * s), 5),
      };
    }
    return row;
  });
}

function residualDiagnostics(resid: number[], fitDf: number) {
  const n = resid.length;
  const lags = [12, 24].filter((l) => l < n / 3 && l > fitDf);
  const lb = lags.length ? ljungBox(resid, Math.max(...lags), fitDf).filter((r) => lags.includes(r.lag)) : [];
  const m = resid.reduce((s, v) => s + v, 0) / n;
  const arch = archLm(
    resid.map((v) => v - m),
    Math.min(12, Math.floor(n / 4)),
  );
  const jb = jarqueBera(resid);
  return {
    raw: { archP: arch.pValue, jbP: jb.pValue, kurtosis: jb.kurtosis },
    report: {
      ljung_box: lb.map((r) => ({ lag: r.lag, p_value: roundP(r.pValue) })),
      arch_lm_p: roundP(arch.pValue),
      jarque_bera_p: roundP(jb.pValue),
      skewness: round(jb.skewness, 3),
      kurtosis: round(jb.kurtosis, 3),
      white_noise: lb.every((r) => r.pValue >= 0.05),
    },
  };
}

function coefTable(c: { name: string; estimate: number; stdErr: number | null; pValue: number | null }[]) {
  return c.map((x) => ({
    name: x.name,
    estimate: round(x.estimate, 5),
    std_err: x.stdErr === null ? null : round(x.stdErr, 4),
    ...(x.pValue === null ? {} : { p_value: roundP(x.pValue) }),
  }));
}

const DEFAULT_ORDERS: number[][] = [
  [0, 1, 0],
  [1, 1, 0],
  [0, 1, 1],
  [1, 1, 1],
  [2, 1, 2],
  [1, 0, 0],
  [2, 0, 0],
  [1, 0, 1],
];

/* ------------------------------ tools ------------------------------ */

export function registerModelingTools(server: McpServer, client: FredClient) {
  server.registerTool(
    "ts_fit_arima",
    {
      title: "Fit and compare ARIMA models",
      description:
        "Fit several ARIMA(p,d,q) candidates by exact maximum likelihood, rank them by AICc, and run residual diagnostics " +
        "(Ljung-Box, ARCH-LM, normality) on each. Flags insignificant terms, near-cancelling AR/MA roots and near unit roots. " +
        "Returns coefficients and an h-step forecast with 80/95% intervals for the best model (or `forecast_order`), " +
        'back-transformed to levels when transform = "log". In-sample fit does not prove forecasting skill: confirm the ' +
        "choice with ts_backtest against a random walk.",
      inputSchema: {
        ...sourceShape,
        transform: levelTransform,
        orders: z
          .array(order)
          .min(1)
          .max(12)
          .optional()
          .describe(
            "Candidate orders. Default: [0,1,0] [1,1,0] [0,1,1] [1,1,1] [2,1,2] [1,0,0] [2,0,0] [1,0,1] (difference-stationary and mean-reverting candidates).",
          ),
        constant: z.boolean().default(true).describe("Include a mean (d=0) or drift (d=1). Default true."),
        forecast_horizon: z.number().int().min(0).max(36).default(6).describe("Steps ahead to forecast. 0 = no forecast. Default 6."),
        forecast_order: order.optional().describe("Order to forecast with. Default: the lowest-AICc candidate."),
      },
      annotations: ANALYSIS,
    },
    safe(async (args) => {
      const s = await loadSeries(client, args, args.transform, 30);
      const y = s.points.map((p) => p.value);
      const orders = args.orders ?? DEFAULT_ORDERS;
      for (const o of orders) if (o[1]! > 2) throw new ToolInputError(`d must be 0, 1 or 2 (got ${o[1]}).`);
      const fits: ArimaFit[] = [];
      const failed: { order: number[]; error: string }[] = [];
      for (const [p, d, q] of orders as [number, number, number][]) {
        try {
          fits.push(fitArima(y, { p, d, q, constant: args.constant }));
        } catch (e) {
          failed.push({ order: [p, d, q], error: e instanceof Error ? e.message : String(e) });
        }
      }
      if (fits.length === 0) throw new Error(`All candidate models failed: ${failed.map((f) => f.error).join("; ")}`);

      // Likelihoods (hence AICc) are only comparable between models with the same d: a d=1 model is
      // fitted to the differences, a d=0 model to the levels. Rank within each d.
      const groups = new Map<number, ArimaFit[]>();
      for (const f of fits) groups.set(f.spec.d, [...(groups.get(f.spec.d) ?? []), f]);
      const h = args.forecast_horizon;
      const dates = futureDates(s, args.frequency, h);
      const describe = (f: ArimaFit, withForecast: boolean) => {
        const diag = residualDiagnostics(f.residuals, f.spec.p + f.spec.q);
        const fc = withForecast && h > 0 ? forecastArima(f, h) : null;
        return {
          order: [f.spec.p, f.spec.d, f.spec.q],
          coefficients: coefTable(f.coefficients),
          ar_inverse_root_moduli: f.arInverseRoots.map(([re, im]) => round(Math.hypot(re, im), 4)),
          ma_inverse_root_moduli: f.maInverseRoots.map(([re, im]) => round(Math.hypot(re, im), 4)),
          residual_diagnostics: diag.report,
          warnings: f.warnings,
          ...(fc ? { forecast: forecastTable(dates, fc.mean, fc.sd, Infinity, s.transform === "log") } : {}),
          _raw: diag.raw,
        };
      };

      const byD = [...groups.entries()]
        .sort(([a], [b]) => a - b)
        .map(([d, group]) => {
          group.sort((a, b) => a.aicc - b.aicc);
          const best = group[0]!;
          return {
            d,
            ranking: group.map((f) => {
              const diag = residualDiagnostics(f.residuals, f.spec.p + f.spec.q);
              return {
                order: [f.spec.p, f.spec.d, f.spec.q],
                aicc: round(f.aicc, 6),
                delta_aicc: round(f.aicc - best.aicc, 4),
                bic: round(f.bic, 6),
                loglik: round(f.loglik, 6),
                residuals_white_noise: diag.report.white_noise,
                residual_ljung_box: diag.report.ljung_box,
                warnings: f.warnings,
              };
            }),
            best,
          };
        });

      let selected: ArimaFit[];
      if (args.forecast_order) {
        const [p, d, q] = args.forecast_order;
        const f = fits.find((x) => x.spec.p === p && x.spec.d === d && x.spec.q === q);
        if (!f) throw new ToolInputError("forecast_order must be one of the fitted orders.");
        selected = [f];
      } else {
        selected = byD.map((g) => g.best);
      }
      const details = selected.map((f) => describe(f, true));
      const anyArch = details.some((d) => d._raw.archP < 0.05);
      const anyNonNormal = details.find((d) => d._raw.jbP < 0.05);

      return ok({
        ...dataNote(s),
        candidates_by_d: byD.map((g) => ({ d: g.d, ranking: g.ranking })),
        ...(failed.length ? { failed } : {}),
        selected: details.map(({ _raw, ...rest }) => rest),
        guidance: [
          "AICc ranks models only within the same d; differences under 2 are negligible, so prefer the simpler model among near-ties.",
          ...(byD.length > 1
            ? ["d=0 (mean-reverting levels) and d=1 (unit root) models are fitted to different data, so their AICc values are not comparable. Decide between them with ts_backtest."]
            : []),
          "Ljung-Box is an adequacy check, not a selection criterion: several models usually pass, and adding terms until one passes overfits.",
          ...(anyArch ? ["Residuals show ARCH effects: consider ts_fit_garch for the variance."] : []),
          ...(anyNonNormal
            ? [`Residuals are non-normal (kurtosis ${anyNonNormal._raw.kurtosis.toFixed(2)}): Gaussian intervals misstate tail risk; compare a Student-t model with ts_fit_garch / ts_backtest.`]
            : []),
        ],
      });
    }),
  );

  server.registerTool(
    "ts_fit_garch",
    {
      title: "Fit AR-GARCH(1,1) volatility model",
      description:
        "Fit a mean model (zero, constant, or AR(p)) with GARCH(1,1) conditional variance and normal or Student-t errors, " +
        "by maximum likelihood with robust standard errors (same conventions as Python's arch package). Reports persistence, " +
        "half-life and unconditional volatility, tests the input for ARCH effects first, checks standardized residuals, and " +
        "compares against the same model with constant variance so you can see whether GARCH is warranted. Use on a " +
        'returns-like series (e.g. transform "log_diff") or on a log level with an AR mean.',
      inputSchema: {
        ...seriesSourceShape,
        mean: z.enum(["zero", "constant", "ar"]).default("constant").describe('Mean model. "ar" uses `ar_lags`. Default "constant".'),
        ar_lags: z.number().int().min(1).max(10).default(1).describe('AR lags when mean = "ar". Default 1.'),
        variance: z.enum(["garch", "constant"]).default("garch").describe('Conditional variance: "garch" = GARCH(1,1). Default "garch".'),
        distribution: z.enum(["normal", "t"]).default("t").describe('Innovation distribution. "t" = standardized Student-t (fat tails). Default "t".'),
        forecast_horizon: z.number().int().min(0).max(36).default(6).describe("Steps ahead to forecast. Default 6."),
      },
      annotations: ANALYSIS,
    },
    safe(async (args) => {
      const s = await loadSeries(client, args, undefined, 40);
      const y = s.points.map((p) => p.value);
      const spec: VolSpec = { mean: args.mean, lags: args.ar_lags, variance: args.variance, dist: args.distribution };

      // Pre-test: ARCH effects in the mean-model residuals.
      const L = spec.mean === "ar" ? spec.lags : 0;
      let pre: number[] = y.slice(L);
      if (spec.mean !== "zero") {
        const X = pre.map((_, t) => [1, ...Array.from({ length: L }, (_, i) => y[t + L - 1 - i]!)]);
        const f = ols(pre, X);
        pre = pre.map((v, t) => v - X[t]!.reduce((u, x, j) => u + x * f.coef[j]!, 0));
      }
      const preArch = archLm(pre, Math.min(12, Math.floor(pre.length / 4)));

      const fit = fitVolModel(y, spec);
      const z = fit.standardizedResiduals;
      const zsq = z.map((v) => v * v);
      const lbZ = ljungBox(z, 12, 0).find((r) => r.lag === 12);
      const lbZ2 = ljungBox(zsq, 12, 0).find((r) => r.lag === 12);
      const archZ = archLm(z, 12);

      let comparison: Record<string, unknown> | undefined;
      if (spec.variance === "garch") {
        const base: VolFit = fitVolModel(y, { ...spec, variance: "constant" });
        const lr = Math.max(0, 2 * (fit.loglik - base.loglik));
        comparison = {
          constant_variance_loglik: round(base.loglik, 6),
          garch_loglik: round(fit.loglik, 6),
          likelihood_ratio: round(lr, 5),
          p_value_chi2_2: roundP(chiSquareSf(lr, 2)),
          delta_aic: round(fit.aic - base.aic, 4),
          delta_bic: round(fit.bic - base.bic, 4),
          conclusion:
            fit.bic < base.bic && chiSquareSf(lr, 2) < 0.05
              ? "GARCH improves the fit: volatility clustering matters for this series."
              : "GARCH does not earn its two extra parameters (BIC and the LR test favor constant variance). Use constant variance; keep the Student-t if tails are fat.",
          note: "alpha = 0 is on the boundary, so the chi2(2) LR p-value is conservative.",
        };
      }

      const h = args.forecast_horizon;
      const fc = h > 0 ? forecastVol(fit, h) : null;
      const vol: Record<string, unknown> = {};
      if (spec.variance === "garch") {
        vol.persistence = round(fit.persistence!, 4);
        vol.half_life_periods = round(fit.halfLife!, 3);
        vol.unconditional_sd = round(Math.sqrt(fit.unconditionalVariance!), 5);
        vol.last_conditional_sd = round(Math.sqrt(fit.conditionalVariance.at(-1)!), 5);
      }
      return ok({
        ...dataNote(s),
        model: modelLabel({ kind: "ar_garch", ar_lags: L, garch: spec.variance === "garch", distribution: spec.dist }),
        pre_test_arch_lm: {
          p_value: roundP(preArch.pValue),
          conclusion: preArch.pValue < 0.05 ? "ARCH effects present in the input." : "No ARCH effects in the input: GARCH is unlikely to help.",
        },
        coefficients: coefTable(fit.coefficients),
        ...(Object.keys(vol).length ? { volatility: vol } : {}),
        loglik: round(fit.loglik, 6),
        aic: round(fit.aic, 6),
        bic: round(fit.bic, 6),
        standardized_residuals: {
          ljung_box_12_p: lbZ ? roundP(lbZ.pValue) : null,
          ljung_box_squared_12_p: lbZ2 ? roundP(lbZ2.pValue) : null,
          arch_lm_12_p: roundP(archZ.pValue),
        },
        ...(comparison ? { garch_vs_constant_variance: comparison } : {}),
        warnings: fit.warnings,
        ...(fc
          ? {
              forecast: forecastTable(
                futureDates(s, args.frequency, h),
                fc.mean,
                fc.sd,
                fit.params.nu,
                s.transform === "log",
              ),
              forecast_note:
                "Intervals use the fitted innovation distribution; for h > 1 with t errors they approximate the predictive as a scaled t.",
            }
          : {}),
      });
    }),
  );

  const modelSpec = z
    .object({
      kind: z
        .enum(["random_walk", "arima", "ar_garch"])
        .describe('"random_walk" (no-change forecast), "arima" (needs `order`), or "ar_garch" (AR mean + optional GARCH(1,1), normal or t).'),
      order: order.optional(),
      constant: z.boolean().optional().describe("arima: include mean/drift. Default true."),
      drift: z.boolean().optional().describe("random_walk: add drift. Default false."),
      ar_lags: z.number().int().min(0).max(5).optional().describe("ar_garch: AR lags (0 = constant mean). Default 1."),
      garch: z.boolean().optional().describe("ar_garch: GARCH(1,1) variance (false = constant variance). Default true."),
      distribution: z.enum(["normal", "t"]).optional().describe('ar_garch: innovation distribution. Default "t".'),
      label: z.string().max(40).optional().describe("Display name. Default derived from the spec."),
    })
    .describe("One model to evaluate.");

  server.registerTool(
    "ts_backtest",
    {
      title: "Rolling-origin forecast backtest",
      description:
        "Out-of-sample comparison of forecasting models by rolling-origin evaluation: re-fit at every origin, forecast " +
        "1..H steps ahead, and score against what actually happened. Reports RMSE relative to a benchmark (default: the " +
        "first model, e.g. a random walk) with a Diebold-Mariano test (HLN-corrected), interval coverage at 80/95%, " +
        "interval score and log predictive score, plus the same comparison split into ex-ante calm vs turbulent periods. " +
        "This is the tool for claims like 'model X beats a random walk'. Runtime grows with origins x models (typically 5-30 s).",
      inputSchema: {
        ...sourceShape,
        transform: levelTransform,
        models: z
          .array(modelSpec)
          .min(2)
          .max(8)
          .describe(
            'Models to compare; the first is the benchmark unless `benchmark` is set. Example: [{"kind":"random_walk"},{"kind":"arima","order":[1,0,0]},{"kind":"ar_garch","ar_lags":1,"garch":false,"distribution":"t"}].',
          ),
        benchmark: z.number().int().min(0).max(7).default(0).describe("Index of the benchmark model in `models`. Default 0."),
        horizons: z.array(z.number().int().min(1).max(24)).min(1).max(6).default([1, 3, 6]).describe("Forecast horizons to score. Default [1, 3, 6]."),
        initial_window: z
          .number()
          .int()
          .min(20)
          .optional()
          .describe("Observations in the first training window. Default: 40% of the sample (at least 30)."),
        step: z.number().int().min(1).max(24).default(1).describe("Origins advance by this many periods. Default 1."),
        window: z.enum(["expanding", "rolling"]).default("expanding").describe("Expanding (all history) or rolling (fixed-length) training window."),
      },
      annotations: ANALYSIS,
    },
    safe(async (args) => {
      const s = await loadSeries(client, args, args.transform, 40);
      const y = s.points.map((p) => p.value);
      const specs: ModelSpec[] = args.models.map((m, i) => {
        if (m.kind === "random_walk") return { kind: "random_walk", drift: m.drift ?? false, label: m.label };
        if (m.kind === "arima") {
          if (!m.order) throw new ToolInputError(`models[${i}]: kind "arima" needs an "order", e.g. [1, 0, 0].`);
          if (m.order[1]! > 2) throw new ToolInputError(`models[${i}]: d must be 0, 1 or 2.`);
          return { kind: "arima", order: m.order as [number, number, number], constant: m.constant ?? true, label: m.label };
        }
        return { kind: "ar_garch", ar_lags: m.ar_lags ?? 1, garch: m.garch ?? true, distribution: m.distribution ?? "t", label: m.label };
      });
      const labels = specs.map(modelLabel);
      const dup = labels.find((l, i) => labels.indexOf(l) !== i);
      if (dup) throw new ToolInputError(`Two models share the label "${dup}"; give one a distinct "label".`);
      if (args.benchmark >= specs.length) throw new ToolInputError("benchmark must index into models.");
      const initial = args.initial_window ?? Math.max(30, Math.floor(0.4 * y.length));
      if (initial >= y.length - Math.min(...args.horizons)) {
        throw new ToolInputError(`initial_window (${initial}) leaves no out-of-sample data; the series has ${y.length} points.`);
      }
      const origins = Math.ceil((y.length - initial) / args.step);
      if (origins * specs.length > 4000) {
        throw new ToolInputError(`${origins} origins x ${specs.length} models is too many fits; raise "step" or drop models.`);
      }
      const t0 = Date.now();
      const r = backtest(y, specs, {
        initial,
        horizons: args.horizons,
        step: args.step,
        window: args.window,
        benchmark: args.benchmark,
      });
      const bench = labels[args.benchmark]!;
      const dateAt = (i: number) => s.points[i]?.date ?? `t${i}`;
      const byHorizon = args.horizons.map((h) => {
        const rows = r.models.map((m) => {
          const sc = m.byHorizon[h]!;
          return {
            model: m.label,
            n: sc.n,
            rel_rmse: round(sc.relRmse, 4),
            rmse: round(sc.rmse, 5),
            mae: round(sc.mae, 5),
            ...(sc.dm ? { dm_stat: round(sc.dm.statistic, 3), dm_p: roundP(sc.dm.pValue) } : {}),
            coverage80: round(sc.coverage80, 3),
            coverage95: round(sc.coverage95, 3),
            interval_score95: round(sc.intervalScore95, 4),
            log_score: round(sc.logScore, 4),
            calm: { rel_rmse: round(sc.calm.relRmse, 4), ...(sc.calm.dm ? { dm_p: roundP(sc.calm.dm.pValue) } : {}) },
            turbulent: { rel_rmse: round(sc.turbulent.relRmse, 4), ...(sc.turbulent.dm ? { dm_p: roundP(sc.turbulent.dm.pValue) } : {}) },
          };
        });
        const others = r.models.filter((_, i) => i !== args.benchmark);
        const bestPoint = [...r.models].sort((a, b) => a.byHorizon[h]!.rmse - b.byHorizon[h]!.rmse)[0]!;
        const bestDensity = [...r.models].sort((a, b) => b.byHorizon[h]!.logScore - a.byHorizon[h]!.logScore)[0]!;
        const sigBetter = others.filter((m) => m.byHorizon[h]!.dm && m.byHorizon[h]!.dm!.pValue < 0.05 && m.byHorizon[h]!.dm!.meanDiff < 0);
        const sigWorse = others.filter((m) => m.byHorizon[h]!.dm && m.byHorizon[h]!.dm!.pValue < 0.05 && m.byHorizon[h]!.dm!.meanDiff > 0);
        return {
          horizon: h,
          models: rows,
          summary: {
            lowest_rmse: bestPoint.label,
            best_log_score: bestDensity.label,
            significantly_better_than_benchmark: sigBetter.map((m) => m.label),
            significantly_worse_than_benchmark: sigWorse.map((m) => m.label),
            verdict: sigBetter.length
              ? `${sigBetter.map((m) => m.label).join(", ")} beat ${bench} at the 5% level (Diebold-Mariano).`
              : `No model beats ${bench} significantly at the 5% level for h=${h}.`,
          },
        };
      });
      return ok({
        ...dataNote(s),
        setup: {
          benchmark: bench,
          window: args.window,
          initial_window: initial,
          origins: r.origins,
          first_origin: dateAt(r.firstOrigin),
          last_origin: dateAt(r.lastOrigin),
          loss_scale: s.transform === "log" ? "log scale (errors are approximately relative errors)" : "original units",
          elapsed_ms: Date.now() - t0,
          ...(Object.keys(r.failures).length ? { failed_fits: r.failures } : {}),
        },
        shock_concentration: {
          top5pct_share_of_benchmark_sq_error: round(r.benchmarkTopShockShare, 3),
          note: "Share of the benchmark's total squared one-step error coming from its largest 5% of errors. High values mean a few shocks dominate the loss, which makes accuracy differences hard to establish.",
        },
        results: byHorizon,
        reading_guide: [
          "rel_rmse < 1 means lower error than the benchmark; dm_p < 0.05 means the difference is statistically significant.",
          "Nominal coverage is 0.80 / 0.95: higher means intervals are too wide, lower means too narrow.",
          "log_score: higher is better; it rewards calibrated intervals and fat tails where they are needed.",
          "calm / turbulent split origins by trailing 12-period volatility (above vs below the median).",
        ],
      });
    }),
  );
}

