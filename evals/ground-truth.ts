/**
 * How the eval labels were derived: runs the validated tools on candidate FRED series (data through
 * 2026-10) and prints the facts the cases rely on (ARCH effects, seasonality, stationarity, backtests).
 *
 *   NODE_USE_ENV_PROXY=1 npx tsx evals/ground-truth.ts
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { FredClient } from "../src/fred/client.js";
import { createServer } from "../src/server.js";
const server = createServer(new FredClient({ apiKey: undefined, csvFallback: true }));
const c = new Client({ name: "facts", version: "0" });
const [a, b] = InMemoryTransport.createLinkedPair();
await Promise.all([server.connect(b), c.connect(a)]);
const call = async (name: string, args: any) => { const r: any = await c.callTool({ name, arguments: args }); if (r.isError) return { error: r.content[0].text }; return JSON.parse(r.content[0].text); };
const END = "2026-10-31";
for (const id of ["GVZCLS", "VIXCLS", "DGS10", "UNRATENSA", "CPIAUCSL", "HOUSTNSA", "INDPRO"]) {
  const m = { series_id: id, frequency: "m", end: END };
  const obs = await call("fred_get_observations", { ...m, max_points: 0 });
  const ac = await call("ts_autocorrelation", { ...m, transform: "log_diff", max_lag: 12 });
  const stl = await call("ts_decompose_stl", { ...m, transform: "log" });
  const st = await call("ts_test_stationarity", { ...m, transforms: ["log", "log_diff"] });
  console.log(id, "n", obs.n, obs.range, "| ARCH m p", ac.arch_effects?.arch_lm.p_value, "| JB kurt", ac.normality?.kurtosis, "| F_S", stl.seasonal_strength, "F_T", stl.trend_strength, "| stat", st.results?.map((r: any) => `${r.transform}:${r.conclusion}`).join(","), "| sig acf", JSON.stringify(ac.significant_acf_lags?.slice(0,6)));
}
for (const id of ["VIXCLS", "GVZCLS", "DGS10"]) {
  const d = await call("ts_autocorrelation", { series_id: id, start: "2010-01-01", end: END, transform: "log_diff", max_lag: 10 });
  console.log(id, "DAILY ARCH p", d.arch_effects?.arch_lm.p_value, d.error ?? "");
}
const bt = async (id: string, transform: string, models: any[]) => {
  const r = await call("ts_backtest", { series_id: id, frequency: "m", end: END, start: "1990-01-01", transform, horizons: [1, 3], models, step: 2 });
  if (r.error) return console.log(id, r.error);
  for (const h of r.results) console.log(id, transform, "h", h.horizon, h.models.map((x: any) => `${x.model} rel=${x.rel_rmse} p=${x.dm_p ?? "-"}`).join(" | "), "|", h.summary.verdict);
};
await bt("CPIAUCSL", "log_diff", [{ kind: "random_walk" }, { kind: "arima", order: [1, 0, 0] }, { kind: "arima", order: [1, 0, 1] }]);
await bt("INDPRO", "log_diff", [{ kind: "random_walk" }, { kind: "arima", order: [1, 0, 0] }]);
await bt("DGS10", "none", [{ kind: "random_walk" }, { kind: "arima", order: [1, 1, 0] }, { kind: "arima", order: [1, 0, 0] }]);
await bt("UNRATE", "none", [{ kind: "random_walk" }, { kind: "arima", order: [1, 1, 0] }]);
await c.close();
