import { describe, expect, it } from "vitest";
import { FredClient } from "../../src/fred/client.js";
import { connect, fakeFetch, syntheticDaily, textOf } from "../helpers.js";

const daily = syntheticDaily();
const csv = ["observation_date,GVZCLS", ...daily.map((o) => `${o.date},${o.value === "." ? "" : o.value}`)].join("\n");

function fredRoute(url: URL) {
  const id = url.searchParams.get("series_id") ?? url.searchParams.get("id");
  if (id && id !== "GVZCLS") {
    if (url.pathname.endsWith("fredgraph.csv")) return { status: 404, body: "<html>Error</html>" };
    return { status: 400, body: { error_code: 400, error_message: "Bad Request.  The series does not exist." } };
  }
  if (url.pathname.endsWith("fredgraph.csv")) return { body: csv };
  if (url.pathname.endsWith("/series/search")) {
    return {
      body: {
        count: 1,
        seriess: [
          {
            id: "GVZCLS",
            title: "CBOE Gold ETF Volatility Index",
            observation_start: "2008-06-03",
            observation_end: "2026-11-20",
            frequency: "Daily, Close",
            frequency_short: "D",
            units: "Index",
            units_short: "Index",
            seasonal_adjustment_short: "NSA",
            last_updated: "2026-11-21",
            popularity: 60,
          },
        ],
      },
    };
  }
  if (url.pathname.endsWith("/series/observations")) return { body: { observations: daily } };
  return { status: 404, body: { error_message: "unknown endpoint" } };
}

async function setup(opts: { noKey?: boolean; csvFallback?: boolean } = {}) {
  const f = fakeFetch(fredRoute);
  const client = await connect(
    new FredClient({ apiKey: opts.noKey ? undefined : "test-key", csvFallback: opts.csvFallback ?? opts.noKey, fetch: f.fetch }),
  );
  return { client, calls: f.calls };
}

const call = async (client: Awaited<ReturnType<typeof setup>>["client"], name: string, args: Record<string, unknown>) => {
  const r = await client.callTool({ name, arguments: args });
  return { r, body: r.isError ? undefined : JSON.parse(textOf(r)), text: textOf(r) };
};

const GVZ_MONTHLY = { series_id: "GVZCLS", frequency: "m" } as const;

describe("MCP server", () => {
  it("lists every tool with a substantive description and read-only annotations", async () => {
    const { client } = await setup();
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual([
      "fred_get_observations",
      "fred_get_series_info",
      "fred_search_series",
      "ts_autocorrelation",
      "ts_backtest",
      "ts_compare_series",
      "ts_decompose_stl",
      "ts_fit_arima",
      "ts_fit_garch",
      "ts_test_stationarity",
    ]);
    for (const t of tools) {
      expect(t.description!.length).toBeGreaterThan(100);
      expect(t.annotations?.readOnlyHint).toBe(true);
      // Every parameter carries a description the model can read.
      for (const [name, prop] of Object.entries((t.inputSchema.properties ?? {}) as Record<string, { description?: string }>)) {
        expect(prop.description, `${t.name}.${name}`).toBeTruthy();
      }
    }
  });

  it("searches series", async () => {
    const { client } = await setup();
    const { body } = await call(client, "fred_search_series", { query: "gold volatility" });
    expect(body.results[0].id).toBe("GVZCLS");
  });

  it("aggregates daily data to complete months locally and truncates output; the key never leaks", async () => {
    const { client, calls } = await setup();
    const { body, text } = await call(client, "fred_get_observations", { series_id: "gvzcls", frequency: "m", max_points: 5 });
    expect(body.points).toHaveLength(5);
    expect(body.points[0][0]).toMatch(/-01$/);
    expect(body.missing_dropped).toBeGreaterThan(0);
    expect(body.incomplete_periods_dropped).toHaveLength(1); // synthetic data end mid-month
    expect(body.note).toMatch(/Showing the last 5/);
    expect(calls[0]!.searchParams.get("series_id")).toBe("GVZCLS");
    expect(text).not.toContain("test-key");
  });

  it("works without an API key through the public CSV endpoint", async () => {
    const { client, calls } = await setup({ noKey: true });
    const { body } = await call(client, "fred_get_observations", { ...GVZ_MONTHLY, max_points: 0 });
    expect(body.n).toBeGreaterThan(100);
    expect(calls[0]!.pathname).toMatch(/fredgraph\.csv$/);
    expect(calls[0]!.searchParams.get("api_key")).toBeNull();
  });

  it("compares transforms in one stationarity call and fetches FRED once", async () => {
    const { client, calls } = await setup();
    const { body } = await call(client, "ts_test_stationarity", { ...GVZ_MONTHLY, transforms: ["log", "log_diff"] });
    expect(body.results).toHaveLength(2);
    expect(body.results.find((x: { transform: string }) => x.transform === "log_diff").conclusion).toBe("stationary");
    expect(calls).toHaveLength(1);
  });

  it("reports ARCH effects and normality in ts_autocorrelation", async () => {
    const { client } = await setup({ noKey: true });
    const { body } = await call(client, "ts_autocorrelation", { ...GVZ_MONTHLY, transform: "log_diff", max_lag: 12 });
    expect(body.acf).toHaveLength(12);
    expect(body.arch_effects.conclusion).toMatch(/ARCH/);
    expect(body.normality).toHaveProperty("kurtosis");
  });

  it("STL: strengths, monthly seasonal effects and dated shocks", async () => {
    const { client } = await setup({ noKey: true });
    const { body } = await call(client, "ts_decompose_stl", { ...GVZ_MONTHLY, transform: "log", top_shocks: 3 });
    expect(body.seasonal_effects.map((e: { position: string }) => e.position)).toEqual([
      "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]);
    expect(body.trend_strength).toBeGreaterThanOrEqual(0);
    expect(body.largest_shocks).toHaveLength(3);
    expect(body.largest_shocks[0].date).toMatch(/^\d{4}-\d{2}-01$/);
  });

  it("ARIMA: ranks within each d and back-transforms log forecasts to dated levels", async () => {
    const { client } = await setup({ noKey: true });
    const { body } = await call(client, "ts_fit_arima", {
      ...GVZ_MONTHLY,
      transform: "log",
      orders: [[0, 1, 0], [1, 1, 1], [1, 0, 0]],
      forecast_horizon: 3,
    });
    expect(body.candidates_by_d.map((g: { d: number }) => g.d)).toEqual([0, 1]);
    expect(body.selected).toHaveLength(2); // best d=0 and best d=1
    const fc = body.selected[0].forecast;
    expect(fc).toHaveLength(3);
    expect(fc[0].date).toMatch(/^\d{4}-\d{2}-01$/);
    expect(fc[0].level.median).toBeCloseTo(Math.exp(fc[0].mean), 2); // mean is rounded to 5 s.f.
    expect(fc[0].level.lo95).toBeLessThan(fc[0].level.median);
    expect(body.guidance.join(" ")).toMatch(/not comparable/);
  });

  it("GARCH: pre-test, robust coefficients and a constant-variance comparison", async () => {
    const { client } = await setup({ noKey: true });
    const { body } = await call(client, "ts_fit_garch", { ...GVZ_MONTHLY, transform: "log_diff", forecast_horizon: 2 });
    expect(body.model).toBe("Const-GARCH(1,1)-t");
    expect(body.coefficients.map((c: { name: string }) => c.name)).toEqual(["mu", "omega", "alpha", "beta", "nu"]);
    expect(body.garch_vs_constant_variance).toHaveProperty("delta_bic");
    expect(body.forecast).toHaveLength(2);
  });

  it("backtest: paired comparison against the benchmark with DM tests and regime split", async () => {
    const { client } = await setup({ noKey: true });
    const { body } = await call(client, "ts_backtest", {
      ...GVZ_MONTHLY,
      transform: "log",
      models: [{ kind: "random_walk" }, { kind: "arima", order: [1, 0, 0] }, { kind: "ar_garch", ar_lags: 1, garch: false }],
      horizons: [1, 3],
      step: 3,
    });
    expect(body.setup.benchmark).toBe("RW");
    expect(body.results).toHaveLength(2);
    const h1 = body.results[0];
    expect(h1.models.map((m: { model: string }) => m.model)).toEqual(["RW", "ARIMA(1,0,0)", "AR(1)-t"]);
    expect(h1.models[0].rel_rmse).toBe(1);
    expect(h1.models[1]).toHaveProperty("dm_p");
    expect(h1.models[1].calm).toHaveProperty("rel_rmse");
    expect(h1.summary.verdict).toMatch(/RW/);
    expect(body.shock_concentration.top5pct_share_of_benchmark_sq_error).toBeGreaterThan(0);
  });

  it("compare series: spurious-correlation warning on levels, clean on changes, consistent alignment", async () => {
    const { client } = await setup({ noKey: true });
    // Two independent random walks: the textbook spurious regression.
    let seed = 11;
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
    const walk = () => Array.from({ length: 300 }).reduce<number[]>((acc) => [...acc, (acc.at(-1) ?? 0) + rand()], []);
    const levels = await call(client, "ts_compare_series", { values_x: walk(), values_y: walk() });
    expect(levels.body.warnings.join(" ")).toMatch(/spurious/);
    expect(levels.body.interpretation).toMatch(/caution/);
    // x leads y by 2 periods: CCF peaks at lag +2 and x Granger-causes y, not the reverse.
    const x = Array.from({ length: 300 }, rand);
    const y = x.map((_, t) => (t >= 2 ? 0.8 * x[t - 2]! : 0) + 0.5 * rand());
    const lead = await call(client, "ts_compare_series", { values_x: x, values_y: y, max_lag: 4, granger_lags: 2 });
    expect(lead.body.lead_lag.strongest.lag).toBe(2);
    expect(lead.body.granger.conclusion).toMatch(/^x helps predict y.*y does not help predict x/);
    expect(lead.body.warnings).toEqual([]);
    // A series against itself: correlation 1, Granger degrades gracefully instead of failing the call.
    const self = await call(client, "ts_compare_series", { series_x: "GVZCLS", series_y: "GVZCLS", frequency: "m", transform: "log_diff", max_lag: 3 });
    expect(self.body.correlation.pearson).toBeCloseTo(1, 6);
    expect(self.body.granger.error).toMatch(/singular/);
    // Exactly periodic inputs: the stationarity check is undetermined, the comparison still succeeds.
    const sine = await call(client, "ts_compare_series", {
      values_x: Array.from({ length: 60 }, (_, i) => Math.sin(i)),
      values_y: Array.from({ length: 60 }, (_, i) => Math.sin(i + 1) + 0.1 * Math.cos(3 * i)),
    });
    expect(sine.r.isError).toBeFalsy();
    expect(sine.body.stationarity.x).toBe("undetermined");
  });

  it("compare series rejects mixed inputs with an actionable message", async () => {
    const { client } = await setup({ noKey: true });
    const { r, text } = await call(client, "ts_compare_series", { series_x: "GVZCLS", values_y: Array.from({ length: 40 }, (_, i) => i) });
    expect(r.isError).toBe(true);
    expect(text).toMatch(/not a mix/);
  });

  it("as_of requests an ALFRED vintage (and needs a key)", async () => {
    const { client, calls } = await setup();
    await call(client, "fred_get_observations", { series_id: "GVZCLS", as_of: "2020-01-15", max_points: 0 });
    expect(calls[0]!.searchParams.get("realtime_start")).toBe("2020-01-15");
    expect(calls[0]!.searchParams.get("realtime_end")).toBe("2020-01-15");
    const noKey = await setup({ noKey: true });
    const { r, text } = await call(noKey.client, "fred_get_observations", { series_id: "GVZCLS", as_of: "2020-01-15" });
    expect(r.isError).toBe(true);
    expect(text).toMatch(/FRED_API_KEY/);
  });

  describe("errors are returned as actionable isError results", () => {
    it("unknown series -> hint to search", async () => {
      const { client } = await setup();
      const { r, text } = await call(client, "fred_get_series_info", { series_id: "GVZ" });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/not_found/);
      expect(text).toMatch(/fred_search_series/);
    });

    it("unknown series without a key -> not_found from the CSV endpoint", async () => {
      const { client } = await setup({ noKey: true });
      const { r, text } = await call(client, "fred_get_observations", { series_id: "NOPE" });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/not_found/);
    });

    it("search without a key -> setup instructions", async () => {
      const { client } = await setup({ noKey: true });
      const { r, text } = await call(client, "fred_search_series", { query: "gold" });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/FRED_API_KEY/);
    });

    it("no key and no CSV opt-in -> data tools ask for a key instead of scraping", async () => {
      const { client, calls } = await setup({ noKey: true, csvFallback: false });
      const { r, text } = await call(client, "fred_get_observations", { series_id: "GVZCLS" });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/missing_api_key/);
      expect(text).toMatch(/FRED_CSV_FALLBACK/);
      expect(calls).toHaveLength(0);
    });

    it("both series_id and values -> explains the choice", async () => {
      const { client } = await setup();
      const { r, text } = await call(client, "ts_test_stationarity", { series_id: "GVZCLS", values: Array.from({ length: 20 }, (_, i) => i) });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/exactly one of/);
    });

    it("log of non-positive values -> suggests another transform", async () => {
      const { client } = await setup();
      const { r, text } = await call(client, "ts_test_stationarity", { values: Array.from({ length: 30 }, (_, i) => i - 5), transform: "log" });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/Cannot take log/);
    });

    it("arima model in a backtest without an order -> says what is missing", async () => {
      const { client } = await setup({ noKey: true });
      const { r, text } = await call(client, "ts_backtest", { ...GVZ_MONTHLY, models: [{ kind: "random_walk" }, { kind: "arima" }] });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/needs an "order"/);
    });

    it("schema violations are reported to the model", async () => {
      const { client } = await setup();
      const { r, text } = await call(client, "fred_get_observations", { series_id: "GVZCLS", start: "2020/01/01" });
      expect(r.isError).toBe(true);
      expect(text).toMatch(/YYYY-MM-DD/);
    });
  });
});

describe("release metadata", () => {
  it("package.json, server.json, plugin.json and the server agree on name and version", async () => {
    const { readFileSync } = await import("node:fs");
    const read = (p: string) => JSON.parse(readFileSync(new URL(`../../${p}`, import.meta.url), "utf8"));
    const pkg = read("package.json");
    const reg = read("server.json");
    const plugin = read("plugin/.claude-plugin/plugin.json");
    const { SERVER_VERSION } = await import("../../src/server.js");
    expect(reg.name).toBe(pkg.mcpName);
    expect(reg.description.length).toBeLessThanOrEqual(100);
    for (const v of [reg.version, reg.packages[0].version, plugin.version, SERVER_VERSION]) expect(v).toBe(pkg.version);
    expect(reg.packages[0].identifier).toBe(pkg.name);
    expect(plugin.mcpServers["fred-timeseries"].args).toContain(`${pkg.name}@${pkg.version}`);
  });
});
