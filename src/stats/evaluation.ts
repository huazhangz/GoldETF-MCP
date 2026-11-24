/** Forecast evaluation: Diebold-Mariano test and interval / density scores. */
import { normalLogPdf, stdTLogPdf, studentTCdf } from "./distributions.js";
import { intervalMultiplier } from "./volatility.js";

export interface PredictiveDist {
  mean: number;
  sd: number;
  /** Degrees of freedom of a standardized-t predictive; Infinity = normal. */
  nu: number;
}

export interface DieboldMariano {
  statistic: number;
  pValue: number;
  /** Mean loss differential (model - benchmark). Negative = model has lower loss. */
  meanDiff: number;
  n: number;
}

/**
 * Diebold-Mariano (1995) test of equal predictive accuracy under squared-error loss, with the
 * Harvey, Leybourne & Newbold (1997) small-sample correction and a t(n-1) reference distribution.
 * The long-run variance uses h-1 autocovariances (rectangular kernel), falling back to Bartlett
 * weights if that estimate is not positive.
 */
export function dieboldMariano(eModel: number[], eBench: number[], h: number): DieboldMariano | null {
  const n = eModel.length;
  if (n < 10 || n !== eBench.length) return null;
  const d = eModel.map((e, i) => e * e - eBench[i]! * eBench[i]!);
  const dbar = d.reduce((s, v) => s + v, 0) / n;
  const gamma = (k: number) => {
    let s = 0;
    for (let t = k; t < n; t++) s += (d[t]! - dbar) * (d[t - k]! - dbar);
    return s / n;
  };
  let lrv = gamma(0);
  for (let k = 1; k < h; k++) lrv += 2 * gamma(k);
  if (!(lrv > 0)) {
    lrv = gamma(0);
    for (let k = 1; k < h; k++) lrv += 2 * (1 - k / h) * gamma(k);
  }
  if (!(lrv > 0)) return null;
  const dm = dbar / Math.sqrt(lrv / n);
  const hln = dm * Math.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n);
  const p = 2 * (1 - studentTCdf(Math.abs(hln), n - 1));
  return { statistic: hln, pValue: p, meanDiff: dbar, n };
}

export function logScore(y: number, f: PredictiveDist): number {
  const v = f.sd * f.sd;
  return Number.isFinite(f.nu) ? stdTLogPdf(y - f.mean, v, f.nu) : normalLogPdf(y - f.mean, v);
}

/** Interval bounds for a central interval at `level` (e.g. 0.95). */
export function interval(f: PredictiveDist, level: number): [number, number] {
  const m = intervalMultiplier(level, f.nu) * f.sd;
  return [f.mean - m, f.mean + m];
}

/** Gneiting & Raftery (2007) interval score (lower is better). */
export function intervalScore(y: number, f: PredictiveDist, level: number): number {
  const [lo, hi] = interval(f, level);
  const a = 1 - level;
  return hi - lo + (2 / a) * Math.max(0, lo - y) + (2 / a) * Math.max(0, y - hi);
}
