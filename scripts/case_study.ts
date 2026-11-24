/**
 * GVZ case study, run end-to-end through the MCP server exactly as an agent would call it.
 *
 *   npx tsx scripts/case_study.ts [--end 2026-04-30] [--out docs/case-study]
 *
 * Uses FRED's keyless CSV endpoint (no API key needed), connects an MCP client to the server
 * in-memory, calls the tools in the order an analyst would, and writes:
 *   <out>/gvz-results.json   every tool call's arguments and full JSON result
 *   <out>/gvz-transcript.md  a readable log of the calls with the key numbers
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { FredClient } from "../src/fred/client.js";
import { createServer } from "../src/server.js";

const argv = process.argv.slice(2);
const opt = (name: string, dflt: string) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1]! : dflt;
};
const END = opt("end", "2026-04-30"); // the course project's sample ended 2026-04-28
const OUT = opt("out", "docs/case-study");

const server = createServer(new FredClient({ apiKey: process.env.FRED_API_KEY }));
const client = new Client({ name: "case-study", version: "1.0.0" });
const [a, b] = InMemoryTransport.createLinkedPair();
await Promise.all([server.connect(b), client.connect(a)]);

type Step = { title: string; tool: string; args: Record<string, unknown>; result: any; ms: number };
const steps: Step[] = [];
async function run(title: string, tool: string, args: Record<string, unknown>) {
  const t0 = Date.now();
  const r = await client.callTool({ name: tool, arguments: args });
  const text = (r.content as { text: string }[])[0]!.text;
  if (r.isError) throw new Error(`${tool} failed: ${text}`);
  const result = JSON.parse(text);
  steps.push({ title, tool, args, result, ms: Date.now() - t0 });
  console.error(`✓ ${title} (${Date.now() - t0} ms)`);
  return result;
}

const M = { series_id: "GVZCLS", frequency: "m", end: END } as const;

await run("Data: monthly averages of daily GVZ", "fred_get_observations", { ...M, max_points: 0 });
await run("STL decomposition of log(GVZ)", "ts_decompose_stl", { ...M, transform: "log" });
await run("Stationarity across transforms", "ts_test_stationarity", {
  ...M,
  transforms: ["none", "log", "log_diff", "log_seasonal_diff", "log_diff_seasonal_diff"],
});
await run("Reproduce R tseries::adf.test settings (trend + fixed 5 lags) on log(GVZ)", "ts_test_stationarity", {
  ...M,
  transform: "log",
  regression: "ct",
  adf_lags: 5,
});
await run("Autocorrelation, ARCH effects and normality of monthly log changes", "ts_autocorrelation", {
  ...M,
  transform: "log_diff",
  max_lag: 24,
});
await run("Same diagnostics on DAILY log changes", "ts_autocorrelation", {
  series_id: "GVZCLS",
  end: END,
  transform: "log_diff",
  max_lag: 24,
});
await run("ARIMA candidates on log(GVZ)", "ts_fit_arima", { ...M, transform: "log", forecast_horizon: 5 });
await run("The course project's model: ARIMA(2,1,2)", "ts_fit_arima", {
  ...M,
  transform: "log",
  orders: [[2, 1, 2]],
  forecast_horizon: 5,
});
await run("GARCH(1,1)-t on monthly log changes", "ts_fit_garch", { ...M, transform: "log_diff", mean: "constant" });
await run("AR(1)-GARCH(1,1)-t on monthly log level", "ts_fit_garch", { ...M, transform: "log", mean: "ar", ar_lags: 1 });
await run("GARCH(1,1)-t on DAILY log changes", "ts_fit_garch", {
  series_id: "GVZCLS",
  end: END,
  transform: "log_diff",
  mean: "constant",
  forecast_horizon: 0,
});
await run("Out-of-sample backtest vs random walk", "ts_backtest", {
  ...M,
  transform: "log",
  initial_window: 72,
  horizons: [1, 3, 6],
  models: [
    { kind: "random_walk" },
    { kind: "random_walk", drift: true },
    { kind: "arima", order: [1, 1, 1] },
    { kind: "arima", order: [2, 1, 2] },
    { kind: "arima", order: [1, 0, 0] },
    { kind: "ar_garch", ar_lags: 1, garch: false, distribution: "t" },
    { kind: "ar_garch", ar_lags: 1, garch: true, distribution: "t" },
  ],
});
await run("Latest data: AR(1)-t forecast from the most recent complete month", "ts_fit_garch", {
  series_id: "GVZCLS",
  frequency: "m",
  transform: "log",
  mean: "ar",
  ar_lags: 1,
  variance: "constant",
  distribution: "t",
  forecast_horizon: 6,
});

// What actually happened after the course project's sample ended: score each model's 5-step forecast.
const realized = await run("Realized values after the sample end", "fred_get_observations", {
  series_id: "GVZCLS",
  frequency: "m",
  start: nextMonth(END),
  max_points: 12,
});
const lastLevel = (await run("Last in-sample level", "fred_get_observations", { ...M, max_points: 1 })).points[0][1] as number;
const byStep = (title: string) => steps.find((x) => x.title === title)!.result;
const sel = (r: any, order: number[]) => r.selected.find((x: any) => x.order.join() === order.join());
const paths: Record<string, number[]> = {
  "RW (no change)": Array(5).fill(lastLevel),
  "ARIMA(2,1,2) (course model)": sel(byStep("The course project's model: ARIMA(2,1,2)"), [2, 1, 2]).forecast.map((f: any) => f.level.median),
  "ARIMA(1,1,1)": sel(byStep("ARIMA candidates on log(GVZ)"), [1, 1, 1]).forecast.map((f: any) => f.level.median),
  "AR(1) on log level": sel(byStep("ARIMA candidates on log(GVZ)"), [1, 0, 0]).forecast.map((f: any) => f.level.median),
  "AR(1)-GARCH(1,1)-t": byStep("AR(1)-GARCH(1,1)-t on monthly log level").forecast.slice(0, 5).map((f: any) => f.level.median),
};
const actual: [string, number][] = realized.points.slice(0, 5);
const realizedCheck = Object.entries(paths).map(([model, fc]) => {
  const e = actual.map(([, a], i) => Math.log(a / fc[i]!));
  return { model, forecasts: fc, log_rmse: Math.sqrt(e.reduce((u, v) => u + v * v, 0) / e.length) };
});

await client.close();

function nextMonth(d: string) {
  const [y, m] = d.split("-").map(Number) as [number, number];
  return new Date(Date.UTC(y, m, 1)).toISOString().slice(0, 10);
}

mkdirSync(OUT, { recursive: true });
writeFileSync(
  join(OUT, "gvz-results.json"),
  JSON.stringify({ generated: new Date().toISOString().slice(0, 10), sample_end: END, steps, realized_check: { actual, models: realizedCheck } }, null, 1),
);

const md: string[] = [
  "# GVZ case study: tool-call transcript",
  "",
  `Generated ${new Date().toISOString().slice(0, 10)} by \`scripts/case_study.ts\` (sample end ${END} unless noted).`,
  "Each step is one MCP tool call; full results are in `gvz-results.json`.",
  "",
];
for (const [i, s] of steps.entries()) {
  md.push(`## ${i + 1}. ${s.title}`, "", "```json", `${s.tool} ${JSON.stringify(s.args)}`, "```", "");
  const r = s.result;
  switch (s.tool) {
    case "fred_get_observations":
      md.push(`n = ${r.n}, ${r.range}; dropped missing = ${r.missing_dropped ?? 0}; partial periods dropped = ${JSON.stringify(r.incomplete_periods_dropped ?? [])}.`);
      break;
    case "ts_decompose_stl":
      md.push(`F_T = ${r.trend_strength}, F_S = ${r.seasonal_strength}. ${r.interpretation}`, "");
      md.push(`Largest shocks: ${r.largest_shocks.slice(0, 6).map((x: any) => `${x.date.slice(0, 7)} (${x.remainder})`).join(", ")}`);
      break;
    case "ts_test_stationarity":
      md.push("| transform | ADF p | KPSS p | verdict |", "|---|---|---|---|");
      for (const x of r.results) md.push(`| ${x.transform} | ${x.adf.p_value} | ${x.kpss.p_value}${x.kpss.p_value_note ? ` (${x.kpss.p_value_note})` : ""} | ${x.conclusion} |`);
      break;
    case "ts_autocorrelation":
      md.push(
        `n = ${r.n}. Significant ACF lags: ${JSON.stringify(r.significant_acf_lags)}; PACF: ${JSON.stringify(r.significant_pacf_lags)}. ${r.white_noise}.`,
        "",
        `ARCH-LM p = ${r.arch_effects.arch_lm.p_value}: ${r.arch_effects.conclusion}`,
        "",
        `Jarque-Bera p = ${r.normality.p_value}, skewness ${r.normality.skewness}, kurtosis ${r.normality.kurtosis}.`,
      );
      break;
    case "ts_fit_arima":
      for (const g of r.candidates_by_d) {
        md.push(`**d = ${g.d}**`, "", "| order | ΔAICc | BIC | residuals white noise | warnings |", "|---|---|---|---|---|");
        for (const x of g.ranking) md.push(`| (${x.order}) | ${x.delta_aicc} | ${x.bic} | ${x.residuals_white_noise} | ${x.warnings.join(" ")} |`);
        md.push("");
      }
      for (const sel of r.selected) {
        md.push(`Selected (${sel.order}): ${sel.coefficients.map((c: any) => `${c.name} = ${c.estimate} (se ${c.std_err}${c.p_value !== undefined ? `, p ${c.p_value}` : ""})`).join("; ")}`, "");
        if (sel.forecast) md.push(`Forecast (level median [95%]): ${sel.forecast.map((f: any) => `${f.date.slice(0, 7)} ${f.level.median} [${f.level.lo95}, ${f.level.hi95}]`).join("; ")}`, "");
      }
      break;
    case "ts_fit_garch":
      md.push(
        `Pre-test ARCH-LM p = ${r.pre_test_arch_lm.p_value}. ${r.coefficients.map((c: any) => `${c.name} = ${c.estimate} (se ${c.std_err})`).join("; ")}.`,
        "",
      );
      if (r.volatility) md.push(`Persistence ${r.volatility.persistence}, half-life ${r.volatility.half_life_periods} periods.`, "");
      if (r.garch_vs_constant_variance)
        md.push(`vs constant variance: LR p = ${r.garch_vs_constant_variance.p_value_chi2_2}, ΔBIC = ${r.garch_vs_constant_variance.delta_bic}. ${r.garch_vs_constant_variance.conclusion}`, "");
      if (r.warnings.length) md.push(`Warnings: ${r.warnings.join(" ")}`, "");
      if (r.forecast?.[0]?.level)
        md.push(`Forecast (level median [95%]): ${r.forecast.map((f: any) => `${f.date.slice(0, 7)} ${f.level.median} [${f.level.lo95}, ${f.level.hi95}]`).join("; ")}`);
      else if (r.forecast) md.push(`Forecast (${r.transform}, mean ± sd): ${r.forecast.map((f: any) => `${f.date.slice(0, 7)} ${f.mean} ± ${f.sd}`).join("; ")}`);
      break;
    case "ts_backtest":
      md.push(
        `${r.setup.origins} origins (${r.setup.first_origin} to ${r.setup.last_origin}), expanding window, ${r.setup.elapsed_ms} ms. Top 5% of months = ${r.shock_concentration.top5pct_share_of_benchmark_sq_error} of RW squared error.`,
        "",
      );
      for (const h of r.results) {
        md.push(`**h = ${h.horizon}** — ${h.summary.verdict}`, "", "| model | rel RMSE | DM p | cov80 | cov95 | log score | calm rel | turbulent rel |", "|---|---|---|---|---|---|---|---|");
        for (const x of h.models) md.push(`| ${x.model} | ${x.rel_rmse} | ${x.dm_p ?? "—"} | ${x.coverage80} | ${x.coverage95} | ${x.log_score} | ${x.calm.rel_rmse} | ${x.turbulent.rel_rmse} |`);
        md.push("");
      }
      break;
  }
  md.push("");
}
md.push(
  "## Realized check: forecasts from the course sample vs what happened",
  "",
  `Last in-sample month (${END.slice(0, 7)}) averaged ${lastLevel}. Actual monthly averages: ${actual.map(([d, v]) => `${d.slice(0, 7)} ${v}`).join(", ")}.`,
  "",
  "| model | 5 forecasts (level median) | log RMSE |",
  "|---|---|---|",
  ...realizedCheck.map((r) => `| ${r.model} | ${r.forecasts.map((v) => v.toFixed(2)).join(", ")} | ${r.log_rmse.toFixed(4)} |`),
  "",
  "One five-month episode is an anecdote, not evidence; the rolling-origin backtest above is the evidence.",
  "",
);
writeFileSync(join(OUT, "gvz-transcript.md"), md.join("\n"));
console.error(`wrote ${OUT}/gvz-results.json and ${OUT}/gvz-transcript.md`);
