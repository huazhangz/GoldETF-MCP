/**
 * Rolling-origin (time-series cross-validation) backtest.
 *
 * At each origin T the model is re-estimated on y[0..T-1] (expanding or rolling window) and
 * forecasts T..T+H-1 are scored against the realized values. Every model sees exactly the same
 * origins, so loss differentials are paired and the Diebold-Mariano test applies.
 */
import { fitArima, forecastArima, type ArimaParams } from "./arima.js";
import { dieboldMariano, interval, intervalScore, logScore, type DieboldMariano, type PredictiveDist } from "./evaluation.js";
import { fitVolModel, forecastVol, type Distribution, type VolParams } from "./volatility.js";

export type ModelSpec =
  | { kind: "random_walk"; drift?: boolean; label?: string }
  | { kind: "arima"; order: [number, number, number]; constant?: boolean; label?: string }
  | { kind: "ar_garch"; ar_lags: number; garch: boolean; distribution: Distribution; label?: string };

export function modelLabel(m: ModelSpec): string {
  if (m.label) return m.label;
  if (m.kind === "random_walk") return m.drift ? "RW+drift" : "RW";
  if (m.kind === "arima") return `ARIMA(${m.order.join(",")})${m.constant === false ? " no const" : ""}`;
  const mean = m.ar_lags > 0 ? `AR(${m.ar_lags})` : "Const";
  return `${mean}${m.garch ? "-GARCH(1,1)" : ""}-${m.distribution === "t" ? "t" : "N"}`;
}

type Warm = ArimaParams | VolParams | undefined;

/** Fit a model on `train` and return predictive distributions for steps 1..h. */
export function fitAndForecast(m: ModelSpec, train: number[], h: number, warm?: Warm): { f: PredictiveDist[]; warm: Warm } {
  if (m.kind === "random_walk" || m.kind === "arima") {
    const [p, d, q] = m.kind === "arima" ? m.order : [0, 1, 0];
    const constant = m.kind === "arima" ? (m.constant ?? true) : Boolean(m.drift);
    const fit = fitArima(train, { p, d, q, constant }, { start: warm as ArimaParams | undefined, computeStdErr: false, fast: true });
    const fc = forecastArima(fit, h);
    return { f: fc.mean.map((mean, i) => ({ mean, sd: fc.sd[i]!, nu: Infinity })), warm: fit.params };
  }
  const fit = fitVolModel(
    train,
    { mean: m.ar_lags > 0 ? "ar" : "constant", lags: m.ar_lags, variance: m.garch ? "garch" : "constant", dist: m.distribution },
    { start: warm as VolParams | undefined, computeStdErr: false, fast: true },
  );
  const fc = forecastVol(fit, h);
  // Multi-step predictive is approximated by a scaled t with the fitted nu (exact for h = 1).
  return { f: fc.mean.map((mean, i) => ({ mean, sd: fc.sd[i]!, nu: fit.params.nu })), warm: fit.params };
}

export interface BacktestOptions {
  /** Number of observations in the first training window. */
  initial: number;
  horizons: number[];
  step?: number;
  window?: "expanding" | "rolling";
  /** Index into models of the benchmark (default 0). */
  benchmark?: number;
  /** Trailing window (in observations) used to classify each origin as a calm or turbulent regime. */
  regimeWindow?: number;
}

export interface HorizonScore {
  n: number;
  rmse: number;
  mae: number;
  relRmse: number;
  dm: DieboldMariano | null;
  coverage80: number;
  coverage95: number;
  intervalScore95: number;
  logScore: number;
  /** Relative RMSE / DM within ex-ante calm and turbulent regimes (split at the median trailing volatility). */
  calm: { n: number; relRmse: number; dm: DieboldMariano | null };
  turbulent: { n: number; relRmse: number; dm: DieboldMariano | null };
}

export interface BacktestResult {
  origins: number;
  firstOrigin: number;
  lastOrigin: number;
  failures: Record<string, number>;
  models: { label: string; byHorizon: Record<number, HorizonScore> }[];
  /** Share of the benchmark's total squared error at h=min(horizons) from its largest 5% of errors. */
  benchmarkTopShockShare: number;
}

interface Row {
  origin: number;
  y: number;
  f: PredictiveDist;
}

const rmse = (e: number[]) => Math.sqrt(e.reduce((s, v) => s + v * v, 0) / e.length);

export function backtest(y: number[], models: ModelSpec[], opts: BacktestOptions): BacktestResult {
  const H = Math.max(...opts.horizons);
  const step = opts.step ?? 1;
  const bench = opts.benchmark ?? 0;
  const regimeWindow = opts.regimeWindow ?? 12;
  if (opts.initial < 20) throw new Error("initial training window must be at least 20 observations.");
  if (opts.initial + Math.min(...opts.horizons) > y.length) {
    throw new Error(`initial (${opts.initial}) leaves no out-of-sample observations (series length ${y.length}).`);
  }

  const origins: number[] = [];
  for (let o = opts.initial; o + Math.min(...opts.horizons) - 1 < y.length; o += step) origins.push(o);

  // Ex-ante regime: trailing std of first differences at each origin.
  const trailingVol = origins.map((o) => {
    const d = [];
    for (let t = Math.max(1, o - regimeWindow); t < o; t++) d.push(y[t]! - y[t - 1]!);
    const m = d.reduce((s, v) => s + v, 0) / d.length;
    return Math.sqrt(d.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, d.length - 1));
  });
  const sortedVol = [...trailingVol].sort((a, b) => a - b);
  const medianVol = sortedVol[Math.floor(sortedVol.length / 2)]!;
  const turbulentOrigin = new Map(origins.map((o, i) => [o, trailingVol[i]! > medianVol]));

  const rows: Map<number, Row[]>[] = models.map(() => new Map(opts.horizons.map((h) => [h, [] as Row[]])));
  const failures: Record<string, number> = {};
  const warm: Warm[] = models.map(() => undefined);
  for (const o of origins) {
    const train = opts.window === "rolling" ? y.slice(o - opts.initial, o) : y.slice(0, o);
    models.forEach((m, mi) => {
      try {
        const out = fitAndForecast(m, train, H, warm[mi]);
        warm[mi] = out.warm;
        for (const h of opts.horizons) {
          const t = o + h - 1;
          if (t < y.length) rows[mi]!.get(h)!.push({ origin: o, y: y[t]!, f: out.f[h - 1]! });
        }
      } catch {
        const label = modelLabel(m);
        failures[label] = (failures[label] ?? 0) + 1;
      }
    });
  }

  // Score only origins where every model produced a forecast (paired comparison).
  const common = new Map(
    opts.horizons.map((h) => {
      const sets = rows.map((r) => new Set(r.get(h)!.map((x) => x.origin)));
      return [h, new Set([...sets[0]!].filter((o) => sets.every((s) => s.has(o))))];
    }),
  );

  const errs = (mi: number, h: number, filter?: (o: number) => boolean) =>
    rows[mi]!.get(h)!
      .filter((r) => common.get(h)!.has(r.origin) && (!filter || filter(r.origin)))
      .map((r) => r.y - r.f.mean);

  const result = models.map((m, mi) => {
    const byHorizon: Record<number, HorizonScore> = {};
    for (const h of opts.horizons) {
      const rs = rows[mi]!.get(h)!.filter((r) => common.get(h)!.has(r.origin));
      const e = rs.map((r) => r.y - r.f.mean);
      const eb = errs(bench, h);
      const regime = (turb: boolean) => {
        const f = (o: number) => turbulentOrigin.get(o) === turb;
        const em = errs(mi, h, f);
        const ebm = errs(bench, h, f);
        return { n: em.length, relRmse: rmse(em) / rmse(ebm), dm: mi === bench ? null : dieboldMariano(em, ebm, h) };
      };
      const cover = (level: number) =>
        rs.filter((r) => {
          const [lo, hi] = interval(r.f, level);
          return r.y >= lo && r.y <= hi;
        }).length / rs.length;
      byHorizon[h] = {
        n: e.length,
        rmse: rmse(e),
        mae: e.reduce((s, v) => s + Math.abs(v), 0) / e.length,
        relRmse: rmse(e) / rmse(eb),
        dm: mi === bench ? null : dieboldMariano(e, eb, h),
        coverage80: cover(0.8),
        coverage95: cover(0.95),
        intervalScore95: rs.reduce((s, r) => s + intervalScore(r.y, r.f, 0.95), 0) / rs.length,
        logScore: rs.reduce((s, r) => s + logScore(r.y, r.f), 0) / rs.length,
        calm: regime(false),
        turbulent: regime(true),
      };
    }
    return { label: modelLabel(m), byHorizon };
  });

  const e1 = errs(bench, Math.min(...opts.horizons)).map((v) => v * v).sort((a, b) => b - a);
  const top = Math.max(1, Math.round(e1.length * 0.05));
  const total = e1.reduce((s, v) => s + v, 0);
  return {
    origins: origins.length,
    firstOrigin: origins[0]!,
    lastOrigin: origins.at(-1)!,
    failures,
    models: result,
    benchmarkTopShockShare: total > 0 ? e1.slice(0, top).reduce((s, v) => s + v, 0) / total : 0,
  };
}
