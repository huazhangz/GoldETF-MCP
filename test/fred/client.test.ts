import { describe, expect, it } from "vitest";
import { FredClient, FredError } from "../../src/fred/client.js";
import { fakeFetch } from "../helpers.js";

const err = (status: number, message: string) => () => ({ status, body: { error_code: status, error_message: message } });

async function kindOf(p: Promise<unknown>): Promise<string> {
  try {
    await p;
    return "none";
  } catch (e) {
    expect(e).toBeInstanceOf(FredError);
    return (e as FredError).kind;
  }
}

describe("FredClient error mapping", () => {
  it.each([
    [err(400, "Bad Request.  The series does not exist."), "not_found"],
    [err(400, "Bad Request.  The value for variable api_key is not registered."), "invalid_api_key"],
    [err(429, "Too Many Requests."), "rate_limited"],
    [err(500, "Internal Server Error"), "upstream"],
    [err(400, "Bad Request.  Variable observation_start is not a valid date."), "bad_request"],
  ])("maps FRED responses to error kinds (%#)", async (route, kind) => {
    const c = new FredClient({ apiKey: "k", fetch: fakeFetch(route).fetch, retryDelayMs: 0 });
    expect(await kindOf(c.getSeries("X"))).toBe(kind);
  });

  it("fails fast without an API key and makes no request", async () => {
    const f = fakeFetch(() => ({ body: {} }));
    const c = new FredClient({ apiKey: undefined, fetch: f.fetch });
    expect(await kindOf(c.getSeries("GVZCLS"))).toBe("missing_api_key");
    expect(f.calls).toHaveLength(0);
  });

  it("wraps network failures", async () => {
    const c = new FredClient({
      apiKey: "k",
      retryDelayMs: 0,
      fetch: (async () => {
        throw new TypeError("fetch failed");
      }) as typeof fetch,
    });
    expect(await kindOf(c.getSeries("GVZCLS"))).toBe("network");
  });

  it("falls back to the keyless CSV endpoint and parses blanks as missing", async () => {
    const f = fakeFetch(() => ({ body: "observation_date,GVZCLS\n2020-01-01,15.2\n2020-01-02,\n2020-01-03,16" }));
    const c = new FredClient({ apiKey: undefined, csvFallback: true, fetch: f.fetch });
    expect(await c.getObservations("GVZCLS", { start: "2020-01-02" })).toEqual([
      { date: "2020-01-02", value: null },
      { date: "2020-01-03", value: 16 },
    ]);
    expect(f.calls[0]!.searchParams.get("id")).toBe("GVZCLS");
  });

  it("parses missing values encoded as '.'", async () => {
    const f = fakeFetch(() => ({
      body: { observations: [{ date: "2020-01-01", value: "15.2" }, { date: "2020-01-02", value: "." }] },
    }));
    const c = new FredClient({ apiKey: "k", fetch: f.fetch });
    expect(await c.getObservations("GVZCLS")).toEqual([
      { date: "2020-01-01", value: 15.2 },
      { date: "2020-01-02", value: null },
    ]);
  });

  it("retries transient failures, then succeeds", async () => {
    let n = 0;
    const f = fakeFetch(() => (++n < 3 ? { status: 503, body: { error_message: "busy" } } : { body: { seriess: [{ id: "GVZCLS" }] } }));
    const c = new FredClient({ apiKey: "k", fetch: f.fetch, retryDelayMs: 0 });
    expect((await c.getSeries("GVZCLS")).id).toBe("GVZCLS");
    expect(f.calls).toHaveLength(3);
  });

  it("does not retry client errors", async () => {
    const f = fakeFetch(() => ({ status: 400, body: { error_message: "Bad Request.  The series does not exist." } }));
    const c = new FredClient({ apiKey: "k", fetch: f.fetch, retryDelayMs: 0 });
    expect(await kindOf(c.getSeries("NOPE"))).toBe("not_found");
    expect(f.calls).toHaveLength(1);
  });

  it("caches responses with an LRU cap", async () => {
    const f = fakeFetch((url) => ({ body: { seriess: [{ id: url.searchParams.get("series_id") }] } }));
    const c = new FredClient({ apiKey: "k", fetch: f.fetch, cacheEntries: 2 });
    await c.getSeries("A");
    await c.getSeries("B");
    await c.getSeries("A"); // hit; A becomes most recent
    await c.getSeries("C"); // evicts B
    await c.getSeries("A"); // still cached
    await c.getSeries("B"); // refetched
    expect(f.calls.map((u) => u.searchParams.get("series_id"))).toEqual(["A", "B", "C", "B"]);
  });
});
