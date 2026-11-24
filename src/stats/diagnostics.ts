import { chiSquareSf } from "./distributions.js";
import { ols } from "./ols.js";

export interface LmTest {
  statistic: number;
  df: number;
  pValue: number;
}

/**
 * Engle's ARCH-LM test: regress e_t^2 on a constant and q of its own lags; LM = n R^2 ~ chi2(q).
 * H0: no conditional heteroskedasticity (no ARCH effects). Same as statsmodels het_arch / R FinTS::ArchTest.
 */
export function archLm(resid: number[], lags: number): LmTest {
  const e2 = resid.map((e) => e * e);
  const y: number[] = [];
  const X: number[][] = [];
  for (let t = lags; t < e2.length; t++) {
    y.push(e2[t]!);
    X.push([1, ...Array.from({ length: lags }, (_, i) => e2[t - 1 - i]!)]);
  }
  if (y.length <= lags + 2) throw new Error(`Series too short for an ARCH-LM test with ${lags} lags.`);
  const f = ols(y, X);
  const m = y.reduce((s, v) => s + v, 0) / y.length;
  const tss = y.reduce((s, v) => s + (v - m) ** 2, 0);
  const r2 = tss > 0 ? 1 - f.ssr / tss : 0;
  const statistic = y.length * r2;
  return { statistic, df: lags, pValue: chiSquareSf(statistic, lags) };
}

export interface JarqueBera extends LmTest {
  skewness: number;
  /** Kurtosis (normal = 3). */
  kurtosis: number;
}

/** Jarque-Bera normality test. H0: skewness 0 and kurtosis 3. */
export function jarqueBera(x: number[]): JarqueBera {
  const n = x.length;
  const m = x.reduce((s, v) => s + v, 0) / n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (const v of x) {
    const d = v - m;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  const skewness = m3 / m2 ** 1.5;
  const kurtosis = m4 / (m2 * m2);
  const statistic = (n / 6) * (skewness ** 2 + (kurtosis - 3) ** 2 / 4);
  return { statistic, df: 2, pValue: chiSquareSf(statistic, 2), skewness, kurtosis };
}
