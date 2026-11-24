/**
 * Deterministic FRED stand-in for evals.
 *
 * - search / series metadata are served from evals/fixtures/catalog.json (real metadata copied from
 *   FRED; no API key needed, and search results don't drift between runs)
 * - observations are real FRED data, downloaded once from the public CSV endpoint and cached in
 *   evals/.cache/fred/ (gitignored: third-party series such as CBOE's are not redistributed)
 *
 * The MCP server sees an ordinary FRED API (it is constructed with a dummy key), so the eval
 * exercises the same code path as a user with FRED_API_KEY set.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { FredClient, type FredSeries } from "../src/fred/client.js";

const HERE = dirname(fileURLToPath(import.meta.url));
export const CACHE_DIR = join(HERE, ".cache", "fred");
type CatalogEntry = FredSeries & { _keywords: string };
export const CATALOG: CatalogEntry[] = JSON.parse(readFileSync(join(HERE, "fixtures", "catalog.json"), "utf8"));

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
const strip = ({ _keywords, ...s }: CatalogEntry): FredSeries => s;

/** Raw CSV rows for a series, downloading on first use. */
export async function csvRows(id: string): Promise<{ date: string; value: number | null }[]> {
  const path = join(CACHE_DIR, `${id}.csv`);
  if (!existsSync(path)) {
    const r = await fetch(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error(`fixture download for ${id} failed: HTTP ${r.status}`);
    mkdirSync(CACHE_DIR, { recursive: true });
    writeFileSync(path, await r.text());
  }
  return readFileSync(path, "utf8")
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((l) => {
      const [date, v] = l.split(",");
      return { date: date!, value: v === undefined || v === "" || v === "." ? null : Number(v) };
    });
}

function search(q: string, limit: number) {
  const toks = q.toLowerCase().split(/[^a-z0-9-]+/).filter((t) => t.length > 1);
  const scored = CATALOG.map((s) => {
    const hay = `${s.id} ${s.title} ${s._keywords}`.toLowerCase();
    return { s, score: toks.filter((t) => hay.includes(t)).length };
  })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || b.s.popularity - a.s.popularity);
  return { count: scored.length, seriess: scored.slice(0, limit).map((x) => strip(x.s)) };
}

const fixtureFetch: typeof fetch = async (input) => {
  const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url);
  const p = url.searchParams;
  const notFound = () => json({ error_code: 400, error_message: "Bad Request.  The series does not exist." }, 400);
  if (url.pathname.endsWith("/series/search")) return json(search(p.get("search_text") ?? "", Number(p.get("limit") ?? 10)));
  const id = (p.get("series_id") ?? "").toUpperCase();
  const entry = CATALOG.find((s) => s.id === id);
  if (url.pathname.endsWith("/series")) return entry ? json({ seriess: [strip(entry)] }) : notFound();
  if (url.pathname.endsWith("/series/observations")) {
    if (!entry) return notFound();
    const start = p.get("observation_start") ?? "0000";
    const end = p.get("observation_end") ?? "9999";
    const rows = (await csvRows(id)).filter((r) => r.date >= start && r.date <= end);
    return json({ observations: rows.map((r) => ({ date: r.date, value: r.value === null ? "." : String(r.value) })) });
  }
  return json({ error_message: "unknown endpoint" }, 404);
};

export function fixtureClient(): FredClient {
  return new FredClient({ apiKey: "eval-fixture-key", fetch: fixtureFetch });
}
