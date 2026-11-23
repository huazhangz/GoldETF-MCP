import { describe, expect, it } from "vitest";
import { FredClient } from "../../src/fred/client.js";
import { connect, fakeFetch, syntheticDaily, textOf } from "../helpers.js";

const daily = syntheticDaily();

/** Aggregate the synthetic daily data to monthly averages, mimicking FRED's frequency=m&aggregation_method=avg. */
function monthly() {
  const groups = new Map<string, number[]>();
  for (const o of daily) {
    if (o.value === ".") continue;
    const k = `${o.date.slice(0, 7)}-01`;
    groups.set(k, [...(groups.get(k) ?? []), Number(o.value)]);
  }
  return [...groups].map(([date, v]) => ({ date, value: (v.reduce((a, b) => a + b, 0) / v.length).toFixed(4) }));
}

function fredRoute(url: URL) {
  const id = url.searchParams.get("series_id");
  if (id && id !== "GVZCLS") {
    return { status: 400, body: { error_code: 400, error_message: "Bad Request.  The series does not exist." } };
  }
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
  if (url.pathname.endsWith("/series/observations")) {
    return { body: { observations: url.searchParams.get("frequency") === "m" ? monthly() : daily } };
  }
  return { status: 404, body: { error_message: "unknown endpoint" } };
}

async function setup(opts: { noKey?: boolean } = {}) {
  const apiKey = opts.noKey ? undefined : "test-key";
  const f = fakeFetch(fredRoute);
  const client = await connect(new FredClient({ apiKey, fetch: f.fetch }));
  return { client, calls: f.calls };
}

describe("MCP server", () => {
  it("lists all tools with descriptions and read-only annotations", async () => {
    const { client } = await setup();
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual([
      "fred_get_observations",
      "fred_get_series_info",
      "fred_search_series",
      "ts_autocorrelation",
      "ts_test_stationarity",
    ]);
    for (const t of tools) {
      expect(t.description!.length).toBeGreaterThan(80);
      expect(t.annotations?.readOnlyHint).toBe(true);
    }
  });

  it("searches series", async () => {
    const { client } = await setup();
    const r = await client.callTool({ name: "fred_search_series", arguments: { query: "gold volatility" } });
    expect(r.isError).toBeFalsy();
    expect(JSON.parse(textOf(r)).results[0].id).toBe("GVZCLS");
  });

  it("fetches monthly-averaged observations, drops missing values and truncates output", async () => {
    const { client, calls } = await setup();
    const r = await client.callTool({
      name: "fred_get_observations",
      arguments: { series_id: "gvzcls", frequency: "m", max_points: 5 },
    });
    const body = JSON.parse(textOf(r));
    expect(body.points).toHaveLength(5);
    expect(body.observations).toBeGreaterThan(30);
    expect(body.note).toMatch(/Showing the last 5/);
    // series_id normalized to upper case; aggregation defaulted; key never leaks into output
    expect(calls[0]!.searchParams.get("series_id")).toBe("GVZCLS");
    expect(calls[0]!.searchParams.get("aggregation_method")).toBe("avg");
    expect(textOf(r)).not.toContain("test-key");
  });

  it("compares transforms in one stationarity call and fetches FRED once", async () => {
    const { client, calls } = await setup();
    const r = await client.callTool({
      name: "ts_test_stationarity",
      arguments: { series_id: "GVZCLS", transforms: ["log", "log_diff"] },
    });
    expect(r.isError).toBeFalsy();
    const body = JSON.parse(textOf(r));
    expect(body.results).toHaveLength(2);
    const logDiff = body.results.find((x: { transform: string }) => x.transform === "log_diff");
    expect(logDiff.conclusion).toBe("stationary");
    expect(calls).toHaveLength(1); // cache hit for the second transform
  });

  it("runs ACF/PACF/Ljung-Box on inline values without an API key", async () => {
    const { client, calls } = await setup({ noKey: true });
    const values = Array.from({ length: 120 }, (_, i) => Math.sin(i * 1.3) + Math.cos(i * 0.7));
    const r = await client.callTool({ name: "ts_autocorrelation", arguments: { values, max_lag: 12 } });
    expect(r.isError).toBeFalsy();
    const body = JSON.parse(textOf(r));
    expect(body.acf).toHaveLength(12);
    expect(body.pacf).toHaveLength(12);
    expect(calls).toHaveLength(0);
  });

  describe("errors are returned as actionable isError results", () => {
    it("unknown series -> hint to search", async () => {
      const { client } = await setup();
      const r = await client.callTool({ name: "fred_get_series_info", arguments: { series_id: "GVZ" } });
      expect(r.isError).toBe(true);
      expect(textOf(r)).toMatch(/not_found/);
      expect(textOf(r)).toMatch(/fred_search_series/);
    });

    it("missing API key -> setup instructions", async () => {
      const { client } = await setup({ noKey: true });
      const r = await client.callTool({ name: "fred_search_series", arguments: { query: "gold" } });
      expect(r.isError).toBe(true);
      expect(textOf(r)).toMatch(/FRED_API_KEY/);
    });

    it("both series_id and values -> explains the choice", async () => {
      const { client } = await setup();
      const r = await client.callTool({
        name: "ts_test_stationarity",
        arguments: { series_id: "GVZCLS", values: Array.from({ length: 20 }, (_, i) => i) },
      });
      expect(r.isError).toBe(true);
      expect(textOf(r)).toMatch(/exactly one of/);
    });

    it("log of non-positive values -> suggests another transform", async () => {
      const { client } = await setup();
      const values = Array.from({ length: 30 }, (_, i) => i - 5);
      const r = await client.callTool({ name: "ts_test_stationarity", arguments: { values, transform: "log" } });
      expect(r.isError).toBe(true);
      expect(textOf(r)).toMatch(/Cannot take log/);
    });

    it("schema violations are reported to the model", async () => {
      const { client } = await setup();
      const r = await client.callTool({
        name: "fred_get_observations",
        arguments: { series_id: "GVZCLS", start: "2020/01/01" },
      });
      expect(r.isError).toBe(true);
      expect(textOf(r)).toMatch(/YYYY-MM-DD/);
    });
  });
});
