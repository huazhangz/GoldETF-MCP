/**
 * Thin, typed wrapper over the FRED REST API (https://fred.stlouisfed.org/docs/api/fred/).
 * - fetch is injectable for tests
 * - every failure becomes a FredError with a machine-readable `kind`
 * - responses are cached in memory for a short TTL so chained tool calls
 *   (get observations -> test stationarity -> autocorrelation) hit FRED once
 * - optional, opt-in fallback for personal use: without an API key, observations can be read from
 *   FRED's public CSV download (fredgraph.csv, the endpoint pandas-datareader uses). Off by default:
 *   distributed builds should use the user's own key, which binds the FRED API terms of use to them.
 */

export interface FredSeries {
  id: string;
  title: string;
  observation_start: string;
  observation_end: string;
  frequency: string;
  frequency_short: string;
  units: string;
  units_short: string;
  seasonal_adjustment_short: string;
  last_updated: string;
  popularity: number;
  notes?: string;
}

export interface Observation {
  date: string;
  value: number | null; // FRED encodes missing values as "."
}

export type FredErrorKind =
  | "missing_api_key"
  | "invalid_api_key"
  | "not_found"
  | "bad_request"
  | "rate_limited"
  | "network"
  | "upstream";

export class FredError extends Error {
  constructor(
    public readonly kind: FredErrorKind,
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "FredError";
  }
}

export interface FredClientOptions {
  apiKey: string | undefined;
  baseUrl?: string;
  csvUrl?: string;
  /** Allow keyless reads of observations from fredgraph.csv. Default false. */
  csvFallback?: boolean;
  fetch?: typeof fetch;
  timeoutMs?: number;
  cacheTtlMs?: number;
}

export class FredClient {
  private readonly apiKey: string | undefined;
  private readonly baseUrl: string;
  private readonly csvUrl: string;
  private readonly csvFallback: boolean;
  private readonly fetchImpl: typeof fetch;
  private readonly timeoutMs: number;
  private readonly cacheTtlMs: number;
  private readonly cache = new Map<string, { at: number; data: unknown }>();

  constructor(opts: FredClientOptions) {
    this.apiKey = opts.apiKey;
    this.baseUrl = opts.baseUrl ?? "https://api.stlouisfed.org/fred";
    this.csvUrl = opts.csvUrl ?? "https://fred.stlouisfed.org/graph/fredgraph.csv";
    this.csvFallback = opts.csvFallback ?? false;
    this.fetchImpl = opts.fetch ?? fetch;
    this.timeoutMs = opts.timeoutMs ?? 15_000;
    this.cacheTtlMs = opts.cacheTtlMs ?? 10 * 60_000;
  }

  async searchSeries(query: string, limit: number): Promise<{ count: number; series: FredSeries[] }> {
    const res = await this.get<{ count: number; seriess: FredSeries[] }>("series/search", {
      search_text: query,
      limit: String(limit),
      order_by: "popularity",
      sort_order: "desc",
    });
    return { count: res.count, series: res.seriess };
  }

  async getSeries(seriesId: string): Promise<FredSeries> {
    const res = await this.get<{ seriess: FredSeries[] }>("series", { series_id: seriesId });
    const s = res.seriess[0];
    if (!s) throw new FredError("not_found", `Series "${seriesId}" does not exist.`);
    return s;
  }

  /** True when search and metadata tools are usable. */
  get hasApiKey(): boolean {
    return Boolean(this.apiKey);
  }

  /** Native-frequency observations, oldest first. Aggregation is done locally by the caller. */
  async getObservations(seriesId: string, opts: { start?: string; end?: string } = {}): Promise<Observation[]> {
    let obs: Observation[];
    if (this.apiKey || !this.csvFallback) {
      const params: Record<string, string> = { series_id: seriesId };
      if (opts.start) params.observation_start = opts.start;
      if (opts.end) params.observation_end = opts.end;
      const res = await this.get<{ observations: { date: string; value: string }[] }>("series/observations", params);
      obs = res.observations.map((o) => ({ date: o.date, value: o.value === "." ? null : Number(o.value) }));
    } else {
      obs = (await this.getCsv(seriesId)).filter(
        (o) => (!opts.start || o.date >= opts.start) && (!opts.end || o.date <= opts.end),
      );
    }
    return obs;
  }

  private async getCsv(seriesId: string): Promise<Observation[]> {
    const url = new URL(this.csvUrl);
    url.searchParams.set("id", seriesId);
    const key = url.toString();
    const hit = this.cache.get(key);
    if (hit && Date.now() - hit.at < this.cacheTtlMs) return hit.data as Observation[];
    const resp = await this.request(url);
    if (resp.status === 404) {
      throw new FredError("not_found", `Series "${seriesId}" does not exist.`, 404);
    }
    if (!resp.ok) throw new FredError(resp.status >= 500 ? "upstream" : "bad_request", `FRED CSV download failed (HTTP ${resp.status}).`, resp.status);
    const text = await resp.text();
    const lines = text.trim().split(/\r?\n/);
    if (!/^observation_date|^DATE/i.test(lines[0] ?? "")) {
      throw new FredError("upstream", "FRED returned an unexpected CSV format.");
    }
    const obs = lines.slice(1).map((line) => {
      const [date, raw] = line.split(",");
      const v = raw === undefined || raw === "" || raw === "." ? null : Number(raw);
      return { date: date!, value: v === null || Number.isNaN(v) ? null : v };
    });
    this.cache.set(key, { at: Date.now(), data: obs });
    return obs;
  }

  private async request(url: URL): Promise<Response> {
    try {
      return await this.fetchImpl(url, { signal: AbortSignal.timeout(this.timeoutMs) });
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      throw new FredError("network", `Could not reach FRED (${reason}). Retry in a moment.`);
    }
  }

  private async get<T>(path: string, params: Record<string, string>): Promise<T> {
    if (!this.apiKey) {
      throw new FredError(
        "missing_api_key",
        "FRED_API_KEY is not set. Get a free key at https://fredaccount.stlouisfed.org/apikeys and add it to the " +
          '"env" block of this server in your MCP client config (personal use without a key: set FRED_CSV_FALLBACK=1 for data-only access).',
      );
    }
    const url = new URL(`${this.baseUrl}/${path}`);
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    const cacheKey = url.toString();
    const hit = this.cache.get(cacheKey);
    if (hit && Date.now() - hit.at < this.cacheTtlMs) return hit.data as T;

    url.searchParams.set("api_key", this.apiKey);
    url.searchParams.set("file_type", "json");

    const resp = await this.request(url);

    let body: unknown;
    try {
      body = await resp.json();
    } catch {
      throw new FredError("upstream", `FRED returned a non-JSON response (HTTP ${resp.status}).`, resp.status);
    }

    if (!resp.ok) throw toFredError(resp.status, body);
    this.cache.set(cacheKey, { at: Date.now(), data: body });
    return body as T;
  }
}

function toFredError(status: number, body: unknown): FredError {
  const raw =
    typeof body === "object" && body !== null && "error_message" in body
      ? String((body as { error_message: unknown }).error_message)
      : `HTTP ${status}`;
  const msg = raw.replace(/\s+/g, " ").trim();
  if (status === 429) return new FredError("rate_limited", "FRED rate limit hit (120 requests/minute).", status);
  if (/api_key/i.test(msg)) return new FredError("invalid_api_key", `FRED rejected the API key: ${msg}`, status);
  if (/does not exist/i.test(msg)) return new FredError("not_found", msg, status);
  if (status >= 500) return new FredError("upstream", `FRED server error: ${msg}`, status);
  return new FredError("bad_request", msg, status);
}
