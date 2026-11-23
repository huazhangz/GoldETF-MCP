import { chiSquareSf } from "./distributions.js";

function mean(x: number[]): number {
  return x.reduce((s, v) => s + v, 0) / x.length;
}

/** Sample autocorrelation for lags 0..maxLag (biased estimator, as in R `acf` and statsmodels). */
export function acf(x: number[], maxLag: number): number[] {
  const n = x.length;
  if (maxLag >= n) throw new RangeError(`maxLag (${maxLag}) must be smaller than the series length (${n}).`);
  const m = mean(x);
  const d = x.map((v) => v - m);
  const c0 = d.reduce((s, v) => s + v * v, 0);
  if (c0 === 0) throw new Error("Series is constant; autocorrelation is undefined.");
  const out: number[] = [1];
  for (let k = 1; k <= maxLag; k++) {
    let ck = 0;
    for (let t = k; t < n; t++) ck += d[t]! * d[t - k]!;
    out.push(ck / c0);
  }
  return out;
}

/** Partial autocorrelation for lags 1..maxLag via Durbin-Levinson recursion on the sample ACF. */
export function pacf(x: number[], maxLag: number): number[] {
  const r = acf(x, maxLag);
  const out: number[] = [];
  let phiPrev: number[] = [];
  for (let k = 1; k <= maxLag; k++) {
    let num = r[k]!;
    let den = 1;
    for (let j = 1; j < k; j++) {
      num -= phiPrev[j - 1]! * r[k - j]!;
      den -= phiPrev[j - 1]! * r[j]!;
    }
    const phiKK = num / den;
    const phi = new Array<number>(k);
    for (let j = 1; j < k; j++) phi[j - 1] = phiPrev[j - 1]! - phiKK * phiPrev[k - j - 1]!;
    phi[k - 1] = phiKK;
    phiPrev = phi;
    out.push(phiKK);
  }
  return out;
}

export interface LjungBoxResult {
  lag: number;
  statistic: number;
  df: number;
  pValue: number;
}

/**
 * Ljung-Box portmanteau test at each lag 1..maxLag.
 * `fitDf` is the number of estimated ARMA parameters (p + q) when testing model residuals;
 * lags with df <= 0 are skipped.
 */
export function ljungBox(x: number[], maxLag: number, fitDf = 0): LjungBoxResult[] {
  const n = x.length;
  const r = acf(x, maxLag);
  const out: LjungBoxResult[] = [];
  let q = 0;
  for (let k = 1; k <= maxLag; k++) {
    q += (r[k]! * r[k]!) / (n - k);
    const df = k - fitDf;
    if (df <= 0) continue;
    const statistic = n * (n + 2) * q;
    out.push({ lag: k, statistic, df, pValue: chiSquareSf(statistic, df) });
  }
  return out;
}
