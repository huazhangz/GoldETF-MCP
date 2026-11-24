/**
 * Ordinary least squares via the normal equations with Gauss-Jordan inversion.
 * Design matrices in this project are tiny (k <= ~20 columns), so this is
 * numerically adequate and keeps the code readable.
 */
import { invert } from "./linalg.js";

export interface OlsResult {
  coef: number[];
  stdErr: number[];
  /** Residual sum of squares. */
  ssr: number;
  nobs: number;
  /** Akaike information criterion, computed as in statsmodels (Gaussian log-likelihood). */
  aic: number;
}

/** Fit y = X b + e. `X` is row-major (one row per observation). */
export function ols(y: number[], X: number[][]): OlsResult {
  const n = y.length;
  const k = X[0]?.length ?? 0;
  if (n !== X.length) throw new Error("ols: y and X row counts differ");
  if (n <= k) throw new Error(`ols: need more observations (${n}) than regressors (${k})`);

  const xtx = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const xty = new Array<number>(k).fill(0);
  for (let i = 0; i < n; i++) {
    const row = X[i]!;
    for (let a = 0; a < k; a++) {
      xty[a]! += row[a]! * y[i]!;
      for (let b = a; b < k; b++) xtx[a]![b]! += row[a]! * row[b]!;
    }
  }
  for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) xtx[a]![b] = xtx[b]![a]!;

  const inv = invert(xtx);
  const coef = inv.map((row) => row.reduce((s, v, j) => s + v * xty[j]!, 0));

  let ssr = 0;
  for (let i = 0; i < n; i++) {
    const fitted = X[i]!.reduce((s, v, j) => s + v * coef[j]!, 0);
    ssr += (y[i]! - fitted) ** 2;
  }
  const sigma2 = ssr / (n - k);
  const stdErr = inv.map((row, i) => Math.sqrt(row[i]! * sigma2));
  const llf = -(n / 2) * (Math.log(2 * Math.PI) + Math.log(ssr / n) + 1);
  const aic = -2 * llf + 2 * k;
  return { coef, stdErr, ssr, nobs: n, aic };
}
