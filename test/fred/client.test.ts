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
    const c = new FredClient({ apiKey: "k", fetch: fakeFetch(route).fetch });
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
      fetch: (async () => {
        throw new TypeError("fetch failed");
      }) as typeof fetch,
    });
    expect(await kindOf(c.getSeries("GVZCLS"))).toBe("network");
  });

  it("falls back to the keyless CSV endpoint and parses blanks as missing", async () => {
    const f = fakeFetch(() => ({ body: "observation_date,GVZCLS\n2020-01-01,15.2\n2020-01-02,\n2020-01-03,16" }));
    const c = new FredClient({ apiKey: undefined, fetch: f.fetch });
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
});
