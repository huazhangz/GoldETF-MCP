import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { FredClient } from "../src/fred/client.js";
import { createServer } from "../src/server.js";

/** Deterministic GVZ-like daily series: log-random-walk around ~18, business days only. */
export function syntheticDaily(start = "2008-06-02", days = 2600, seed = 7): { date: string; value: string }[] {
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) % 2 ** 31;
    return s / 2 ** 31 - 0.5;
  };
  const out: { date: string; value: string }[] = [];
  const d = new Date(`${start}T00:00:00Z`);
  let logv = Math.log(18);
  while (out.length < days) {
    const dow = d.getUTCDay();
    if (dow !== 0 && dow !== 6) {
      // mean-reverting log level, like an implied-volatility index
      logv += 0.01 * (Math.log(18) - logv) + 0.06 * rand();
      // sprinkle in a FRED-style missing value
      out.push({ date: d.toISOString().slice(0, 10), value: out.length % 97 === 50 ? "." : Math.exp(logv).toFixed(2) });
    }
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return out;
}

export type Route = (url: URL) => { status?: number; body: unknown };

export function fakeFetch(route: Route): { fetch: typeof fetch; calls: URL[] } {
  const calls: URL[] = [];
  const impl = (async (input: string | URL | Request) => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url);
    calls.push(url);
    const { status = 200, body } = route(url);
    if (typeof body === "string") return new Response(body, { status, headers: { "content-type": "text/csv" } });
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  }) as typeof fetch;
  return { fetch: impl, calls };
}

export async function connect(fredClient: FredClient) {
  const server = createServer(fredClient);
  const [clientT, serverT] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test", version: "0.0.0" });
  await Promise.all([server.connect(serverT), client.connect(clientT)]);
  return client;
}

export function textOf(result: unknown): string {
  const r = result as { content: { type: string; text: string }[] };
  return r.content.map((c) => c.text).join("\n");
}
