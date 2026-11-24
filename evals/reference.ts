/**
 * Scripted agents used to validate the graders (eval-audit "known-good / known-bad" check):
 *
 *   oracle - the tool calls and final answer a careful analyst would produce; must score ~100%
 *   naive  - plausible-looking but wrong: ignores guardrails, over-claims, guesses numbers;
 *            must fail the cases it targets
 *   null   - no tools, "I don't know."; must score ~0%
 *
 * Answers are built from the actual tool results of the same run, never hard-coded numbers.
 */
type Call = { name: string; args: Record<string, unknown> };
type R = any[]; // parsed JSON results of the calls, in order (null for failed calls)

export interface Script {
  calls: Call[];
  answer: (results: R) => string;
}

const M = (series_id: string, end = "2026-10-31") => ({ series_id, frequency: "m", end });
const anySig = (bt: any) => bt.results.some((h: any) => h.summary.significantly_better_than_benchmark.length > 0);
const fc = (res: any) => (res.selected?.[0]?.forecast ?? res.forecast) as any[];

export const ORACLE: Record<string, Script> = {
  garch_gvz_monthly: {
    calls: [{ name: "ts_fit_garch", args: { ...M("GVZCLS"), transform: "log_diff" } }],
    answer: ([g]) =>
      `GARCH is not warranted for monthly GVZ: the ARCH-LM pre-test p-value is ${g.pre_test_arch_lm.p_value}, alpha is essentially zero (on its lower bound), so beta is not identified and "persistence" is meaningless. ` +
      `BIC prefers constant variance (ΔBIC ${g.garch_vs_constant_variance.delta_bic}). Use constant variance with Student-t errors; volatility clustering lives in daily data.`,
  },
  garch_vix_daily: {
    calls: [{ name: "ts_fit_garch", args: { series_id: "VIXCLS", start: "2010-01-01", end: "2026-10-31", transform: "log_diff" } }],
    answer: ([g]) => {
      const a = g.coefficients.find((c: any) => c.name === "alpha").estimate;
      const b = g.coefficients.find((c: any) => c.name === "beta").estimate;
      return `Yes. ARCH effects are present (ARCH-LM p ${g.pre_test_arch_lm.p_value}). A GARCH(1,1)-t fit gives alpha ${a.toFixed(3)} and beta ${b.toFixed(3)}, so persistence is ${g.volatility.persistence} with a half-life of about ${g.volatility.half_life_periods} trading days.`;
    },
  },
  beat_rw_gvz: {
    calls: [
      {
        name: "ts_backtest",
        args: { ...M("GVZCLS"), transform: "log", horizons: [1, 3], models: [{ kind: "random_walk" }, { kind: "arima", order: [1, 0, 0] }, { kind: "arima", order: [1, 1, 1] }] },
      },
    ],
    answer: ([bt]) =>
      anySig(bt)
        ? `Yes: ${bt.results.map((h: any) => h.summary.verdict).join(" ")}`
        : `No. In a rolling backtest no model is significantly better than the random walk: ${bt.results.map((h: any) => h.summary.verdict).join(" ")}`,
  },
  beat_rw_dgs10: {
    calls: [
      {
        name: "ts_backtest",
        args: { series_id: "DGS10", frequency: "m", start: "1990-01-01", end: "2026-10-31", horizons: [1], models: [{ kind: "random_walk" }, { kind: "arima", order: [1, 1, 0] }, { kind: "arima", order: [1, 0, 0] }] },
      },
    ],
    answer: ([bt]) =>
      anySig(bt) ? `Yes: ${bt.results[0].summary.verdict}` : `No. No model is significantly better than the random walk one month ahead (${bt.results[0].summary.verdict})`,
  },
  beat_rw_cpi: {
    calls: [
      {
        name: "ts_backtest",
        args: { series_id: "CPIAUCSL", frequency: "m", start: "1990-01-01", end: "2026-10-31", transform: "log_diff", horizons: [1, 3], models: [{ kind: "random_walk" }, { kind: "arima", order: [1, 0, 0] }, { kind: "arima", order: [1, 0, 1] }] },
      },
    ],
    answer: ([bt]) =>
      anySig(bt)
        ? `Yes. ${bt.results.map((h: any) => `At h=${h.horizon}: ${h.summary.verdict}`).join(" ")} The AR models are significantly better than the no-change forecast.`
        : `No. No model is significantly better: ${bt.results.map((h: any) => h.summary.verdict).join(" ")}`,
  },
  outlier_unrate_2020: {
    calls: [{ name: "ts_fit_arima", args: { series_id: "UNRATE", end: "2020-04-30", orders: [[1, 1, 0], [0, 1, 1], [1, 0, 0]], forecast_horizon: 1 } }],
    answer: ([a]) =>
      `Caution: April 2020 is an extreme shock (the Covid layoffs), and the warnings show it dominates the Gaussian fit. The d=1 model forecasts ${a.selected.find((s: any) => s.order[1] === 1)?.forecast?.[0]?.mean} for May 2020, which is driven by that single outlier and is unreliable; compare against a random walk or a Student-t model.`,
  },
  stl_houst: {
    calls: [{ name: "ts_decompose_stl", args: { ...M("HOUSTNSA"), transform: "log" } }],
    answer: ([s]) => {
      const top = [...s.seasonal_effects].sort((a: any, b: any) => b.effect - a.effect).slice(0, 3).map((e: any) => e.position);
      return `Yes, housing starts are strongly seasonal: STL seasonal strength F_S = ${s.seasonal_strength}. The highest months are ${top.join(", ")}.`;
    },
  },
  stl_gvz: {
    calls: [{ name: "ts_decompose_stl", args: { ...M("GVZCLS"), transform: "log" } }],
    answer: ([s]) => `No. Seasonality is negligible: F_S = ${s.seasonal_strength}, far below the 0.6 "strong" threshold.`,
  },
  stationarity_dgs10: {
    calls: [{ name: "ts_test_stationarity", args: { ...M("DGS10"), transforms: ["none", "diff"] } }],
    answer: ([t]) =>
      `The level is non-stationary (${t.results[0].conclusion}: ADF p ${t.results[0].adf.p_value}); the first difference is ${t.results[1].conclusion}. Use first differences (d = 1).`,
  },
  stationarity_vix: {
    calls: [{ name: "ts_test_stationarity", args: { ...M("VIXCLS"), transforms: ["log", "log_diff"] } }],
    answer: ([t]) => `VIX is mean-reverting: the log level is ${t.results[0].conclusion} (ADF p ${t.results[0].adf.p_value}), so there is no need to difference it.`,
  },
  quarterly_vix_2024: {
    calls: [{ name: "fred_get_observations", args: { series_id: "VIXCLS", start: "2024-01-01", end: "2024-12-31", frequency: "q", max_points: 10 } }],
    answer: ([o]) => `Quarterly average VIX in 2024: ${o.points.map((p: any) => `${p[0]}: ${p[1].toFixed(2)}`).join(", ")}.`,
  },
  partial_month_gvz: {
    calls: [{ name: "fred_get_observations", args: { series_id: "GVZCLS", start: "2026-11-01", frequency: "m", keep_incomplete_periods: true, max_points: 5 } }],
    answer: ([o]) => `So far in November 2026 (partial month, data through the latest available day) GVZ has averaged ${o.points.at(-1)[1].toFixed(2)}.`,
  },
  cpi_yoy_sep2026: {
    calls: [{ name: "fred_get_observations", args: { series_id: "CPIAUCSL", start: "2025-09-01", end: "2026-09-30", max_points: 20 } }],
    answer: ([o]) => {
      const a = o.points.find((p: any) => p[0] === "2026-09-01")[1];
      const b = o.points.find((p: any) => p[0] === "2025-09-01")[1];
      return `Year-over-year CPI inflation in September 2026 was ${((a / b - 1) * 100).toFixed(2)}%.`;
    },
  },
  discover_gvz: {
    calls: [
      { name: "fred_search_series", args: { query: "gold ETF volatility index" } },
      { name: "fred_get_series_info", args: { series_id: "GVZCLS" } },
    ],
    answer: ([, i]) => `It is ${i.id} (${i.title}), native frequency ${i.frequency}, with data starting ${i.coverage.slice(0, 10)}.`,
  },
  wrong_id_recovery: {
    calls: [
      { name: "ts_test_stationarity", args: { series_id: "GVZ", frequency: "m" } },
      { name: "fred_search_series", args: { query: "GVZ gold volatility" } },
      { name: "ts_test_stationarity", args: { series_id: "GVZCLS", frequency: "m", transforms: ["log", "log_diff"] } },
    ],
    answer: ([, , t]) => `GVZ is not a FRED ID; the gold ETF volatility index is GVZCLS. Results: ${t.results.map((x: any) => `${x.transform}: ${x.conclusion}`).join("; ")}.`,
  },
  arima_choice_gvz: {
    calls: [{ name: "ts_fit_arima", args: { ...M("GVZCLS", "2026-04-30"), transform: "log" } }],
    answer: () =>
      "I would use ARIMA(1,0,0), an AR(1) on the log level: GVZ is mean-reverting. The d=1 candidates are over-differenced (MA roots near 1 cancel the difference) and ARIMA(2,1,2) has insignificant terms.",
  },
  vix_forecast: {
    calls: [{ name: "ts_fit_garch", args: { ...M("VIXCLS"), transform: "log", mean: "ar", ar_lags: 1, variance: "constant", distribution: "t", forecast_horizon: 6 } }],
    answer: ([g]) =>
      `AR(1) on log VIX with Student-t errors (fat tails). ${fc(g)
        .map((f: any) => `${f.date.slice(0, 7)}: ${f.level.median} [95%: ${f.level.lo95} to ${f.level.hi95}]`)
        .join("; ")}.`,
  },
  correlation_out_of_scope: {
    calls: [],
    answer: () =>
      "I can't compute that with the available tools: this server has no correlation function, and copying hundreds of values into a calculation by hand would be unreliable. Fetch both series (GVZCLS, VIXCLS) and compute it in Python or a spreadsheet.",
  },
  concept_no_tools: {
    calls: [],
    answer: () => "ADF: the null hypothesis is that the series has a unit root (non-stationary). KPSS: the null hypothesis is that the series is stationary.",
  },
  efficiency_cpi_acf: {
    calls: [{ name: "ts_autocorrelation", args: { series_id: "CPIAUCSL", frequency: "m", start: "1990-01-01", end: "2026-10-31", transform: "log_diff", max_lag: 12 } }],
    answer: ([a]) => `Yes. Significant autocorrelation at lags ${a.significant_acf_lags.join(", ")}; ${a.white_noise}.`,
  },
};

/** Plausible-but-wrong behaviour: the failure each guardrail or check is meant to catch. */
export const NAIVE: Record<string, Script> = {
  garch_gvz_monthly: {
    calls: [{ name: "ts_fit_garch", args: { ...M("GVZCLS"), transform: "log_diff" } }],
    answer: ([g]) => `The GARCH(1,1) fit gives alpha 0.00 and beta ${g.coefficients.find((c: any) => c.name === "beta").estimate.toFixed(2)}, so volatility is highly persistent (persistence ${g.volatility.persistence}).`,
  },
  garch_vix_daily: {
    calls: [{ name: "ts_autocorrelation", args: { ...M("VIXCLS"), transform: "log_diff" } }],
    answer: () => "No, there is no volatility clustering in VIX.",
  },
  beat_rw_gvz: {
    calls: [{ name: "ts_fit_arima", args: { ...M("GVZCLS"), transform: "log" } }],
    answer: () => "Yes. ARIMA(1,0,0) fits better than the random walk (lower AICc), so it beats the random walk.",
  },
  beat_rw_dgs10: {
    calls: [{ name: "ts_backtest", args: { series_id: "DGS10", frequency: "m", start: "1990-01-01", end: "2026-10-31", horizons: [1], models: [{ kind: "random_walk" }, { kind: "arima", order: [1, 0, 0] }] } }],
    answer: () => "Yes, ARIMA(1,0,0) beats the random walk because its RMSE is lower.",
  },
  beat_rw_cpi: {
    calls: [{ name: "ts_backtest", args: { series_id: "CPIAUCSL", frequency: "m", start: "1990-01-01", end: "2026-10-31", horizons: [1, 3], models: [{ kind: "random_walk" }, { kind: "arima", order: [1, 1, 0] }] } }],
    answer: () => "No model is significantly better than the random walk.",
  },
  outlier_unrate_2020: {
    calls: [{ name: "ts_fit_arima", args: { series_id: "UNRATE", end: "2020-04-30", orders: [[1, 1, 0]], forecast_horizon: 1 } }],
    answer: ([a]) => `ARIMA(1,1,0) forecasts a May 2020 unemployment rate of ${a.selected[0].forecast[0].mean}%.`,
  },
  stl_houst: {
    calls: [{ name: "fred_get_observations", args: { ...M("HOUSTNSA"), max_points: 0 } }],
    answer: () => "Housing starts vary over the year but I'd call the seasonality moderate.",
  },
  stl_gvz: {
    calls: [{ name: "ts_decompose_stl", args: { ...M("GVZCLS"), transform: "log" } }],
    answer: () => "Yes: volatility is highest in October and November, so there is a seasonal pattern worth modeling with SARIMA.",
  },
  stationarity_dgs10: {
    calls: [{ name: "ts_test_stationarity", args: { ...M("DGS10"), transform: "none" } }],
    answer: () => "Yes, the 10-year yield is stationary; you can model it in levels.",
  },
  stationarity_vix: {
    calls: [{ name: "ts_test_stationarity", args: { ...M("VIXCLS"), transform: "none" } }],
    answer: () => "VIX has a unit root, so you must difference it before modeling.",
  },
  quarterly_vix_2024: {
    calls: [{ name: "fred_get_observations", args: { series_id: "VIXCLS", start: "2024-01-01", end: "2024-12-31", frequency: "a", max_points: 5 } }],
    answer: ([o]) => `The VIX averaged about ${o.points[0][1].toFixed(1)} in every quarter of 2024.`,
  },
  partial_month_gvz: {
    calls: [{ name: "fred_get_observations", args: { series_id: "GVZCLS", frequency: "m", max_points: 1 } }],
    answer: ([o]) => `The latest monthly average is ${o.points[0][1].toFixed(2)}.`,
  },
  cpi_yoy_sep2026: {
    calls: [{ name: "fred_get_observations", args: { series_id: "CPIAUCSL", start: "2026-08-01", end: "2026-09-30", transform: "log_diff", max_points: 5 } }],
    answer: ([o]) => `CPI inflation in September 2026 was ${(o.points.at(-1)[1] * 100).toFixed(2)}%.`,
  },
  discover_gvz: { calls: [], answer: () => "The series is GVZ, a monthly index starting in 2005." },
  wrong_id_recovery: {
    calls: [{ name: "ts_test_stationarity", args: { series_id: "GVZ", frequency: "m" } }],
    answer: () => "The series GVZ does not exist on FRED, so I could not run the tests.",
  },
  arima_choice_gvz: {
    calls: [{ name: "ts_fit_arima", args: { ...M("GVZCLS", "2026-04-30"), transform: "log", orders: [[2, 1, 2]] } }],
    answer: () => "I'd use ARIMA(2,1,2) because it is the only model whose residuals pass the Ljung-Box test.",
  },
  vix_forecast: {
    calls: [{ name: "ts_fit_arima", args: { ...M("VIXCLS"), orders: [[1, 0, 0]], forecast_horizon: 6 } }],
    answer: () => "VIX should stay around 18 over the next six months, with a 95% interval of roughly 12 to 25.",
  },
  correlation_out_of_scope: { calls: [], answer: () => "The correlation is about 0.65, so the two move together." },
  concept_no_tools: {
    calls: [{ name: "ts_test_stationarity", args: { ...M("GVZCLS"), transform: "log" } }],
    answer: () => "ADF tests the null of a unit root; KPSS tests the null of stationarity.",
  },
  efficiency_cpi_acf: {
    calls: [{ name: "fred_get_observations", args: { series_id: "CPIAUCSL", frequency: "m", start: "1990-01-01", end: "2026-10-31", transform: "log_diff", max_points: 500 } }],
    answer: () => "Yes, they look autocorrelated.",
  },
};
