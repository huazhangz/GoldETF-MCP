/**
 * Relationships between two series: correlation with an autocorrelation-adjusted significance test,
 * lead/lag cross-correlation, Granger causality, and an OLS slope with Newey-West (HAC) errors.
 */
import { acf } from "./autocorrelation.js";
import { betaInc, normalCdf, studentTCdf } from "./distributions.js";
import { invert } from "./linalg.js";
import { ols } from "./ols.js";

const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length;

export function pearson(x: number[], y: number[]): number {
  const mx = mean(x);
  const my = mean(y);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < x.length; i++) {
    const a = x[i]! - mx;
    const b = y[i]! - my;
    sxy += a * b;
    sxx += a * a;
    syy += b * b;
  }
  if (sxx === 0 || syy === 0) throw new Error("Correlation is undefined for a constant series.");
  return sxy / Math.sqrt(sxx * syy);
}

/** Average ranks (ties share the mean rank), as scipy.stats.rankdata. */
function ranks(x: number[]): number[] {
  const idx = x.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const r = new Array<number>(x.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1]![0] === idx[i]![0]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k]![1]] = avg;
    i = j + 1;
  }
  return r;
}

export function spearman(x: number[], y: number[]): number {
  return pearson(ranks(x), ranks(y));
}

/** Two-sided p-value of H0: rho = 0 using the t statistic with df = n - 2. */
export function correlationPValue(r: number, n: number): number {
  if (n <= 3) return 1;
  const t = (r * Math.sqrt(n - 2)) / Math.sqrt(Math.max(1e-300, 1 - r * r));
  return 2 * (1 - studentTCdf(Math.abs(t), n - 2));
}

/** 95% confidence interval via the Fisher z transform with sample size n. */
export function correlationCI(r: number, n: number): [number, number] {
  const z = Math.atanh(Math.max(-0.999999, Math.min(0.999999, r)));
  const se = 1 / Math.sqrt(Math.max(1, n - 3));
  return [Math.tanh(z - 1.96 * se), Math.tanh(z + 1.96 * se)];
}

/**
 * Effective sample size for the correlation of two autocorrelated series (Bartlett 1946;
 * Pyper & Peterman 1998): n / (1 + 2 sum_k rho_x(k) rho_y(k)), summed to lag n/5.
 */
export function effectiveSampleSize(x: number[], y: number[]): number {
  const n = x.length;
  const K = Math.max(1, Math.min(Math.floor(n / 5), n - 2));
  const ax = acf(x, K);
  const ay = acf(y, K);
  let s = 0;
  for (let k = 1; k <= K; k++) s += ax[k]! * ay[k]!;
  return Math.max(3, Math.min(n, n / (1 + 2 * s)));
}

/** corr(x_t, y_{t+k}) for k = -maxLag..maxLag. Positive k: x leads y by k periods. */
export function crossCorrelation(x: number[], y: number[], maxLag: number): { lag: number; r: number }[] {
  const out: { lag: number; r: number }[] = [];
  for (let k = -maxLag; k <= maxLag; k++) {
    const xs = k >= 0 ? x.slice(0, x.length - k) : x.slice(-k);
    const ys = k >= 0 ? y.slice(k) : y.slice(0, y.length + k);
    out.push({ lag: k, r: pearson(xs, ys) });
  }
  return out;
}

/** Survival function of the F distribution. */
export function fSf(f: number, d1: number, d2: number): number {
  if (!(f > 0)) return 1;
  return betaInc(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
}

export interface GrangerResult {
  lags: number;
  F: number;
  df1: number;
  df2: number;
  pValue: number;
}

/** Does x Granger-cause y? F-test of x's lags in y_t = c + sum a_i y_{t-i} + sum b_i x_{t-i} (statsmodels ssr_ftest). */
export function grangerTest(y: number[], x: number[], p: number): GrangerResult {
  const n = y.length;
  const rows = n - p;
  if (rows - 2 * p - 1 < 5) throw new Error(`Too few observations (${n}) for a Granger test with ${p} lags.`);
  const dep: number[] = [];
  const Xr: number[][] = [];
  const Xu: number[][] = [];
  for (let t = p; t < n; t++) {
    dep.push(y[t]!);
    const own = Array.from({ length: p }, (_, i) => y[t - 1 - i]!);
    const other = Array.from({ length: p }, (_, i) => x[t - 1 - i]!);
    Xr.push([1, ...own]);
    Xu.push([1, ...own, ...other]);
  }
  const r = ols(dep, Xr);
  const u = ols(dep, Xu);
  const df1 = p;
  const df2 = rows - 2 * p - 1;
  const F = (r.ssr - u.ssr) / df1 / (u.ssr / df2);
  return { lags: p, F, df1, df2, pValue: fSf(F, df1, df2) };
}

export interface HacRegression {
  alpha: number;
  beta: number;
  seBeta: number;
  pValue: number;
  r2: number;
  lags: number;
}

/**
 * OLS of y on a constant and x with Newey-West (Bartlett kernel) standard errors, matching statsmodels'
 * cov_type="HAC" defaults (no small-sample correction).
 */
export function hacRegression(y: number[], x: number[], lags?: number): HacRegression {
  const n = y.length;
  const L = lags ?? Math.floor(4 * (n / 100) ** (2 / 9));
  const X = x.map((v) => [1, v]);
  const fit = ols(y, X);
  const e = y.map((v, i) => v - fit.coef[0]! - fit.coef[1]! * x[i]!);
  const XtXinv = invert([
    [n, x.reduce((s, v) => s + v, 0)],
    [x.reduce((s, v) => s + v, 0), x.reduce((s, v) => s + v * v, 0)],
  ]);
  // S = sum_t u_t u_t' + sum_l w_l sum_t (u_t u_{t-l}' + u_{t-l} u_t'), u_t = x_t e_t
  const u = X.map((row, t) => row.map((v) => v * e[t]!));
  const S = [
    [0, 0],
    [0, 0],
  ];
  const add = (a: number[], b: number[], w: number) => {
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) S[i]![j]! += w * a[i]! * b[j]!;
  };
  for (let t = 0; t < n; t++) add(u[t]!, u[t]!, 1);
  for (let l = 1; l <= L; l++) {
    const w = 1 - l / (L + 1);
    for (let t = l; t < n; t++) {
      add(u[t]!, u[t - l]!, w);
      add(u[t - l]!, u[t]!, w);
    }
  }
  const tmp = XtXinv.map((row) => [0, 1].map((j) => row[0]! * S[0]![j]! + row[1]! * S[1]![j]!));
  const V = tmp.map((row) => [0, 1].map((j) => row[0]! * XtXinv[0]![j]! + row[1]! * XtXinv[1]![j]!));
  const seBeta = Math.sqrt(V[1]![1]!);
  const my = mean(y);
  const tss = y.reduce((s, v) => s + (v - my) ** 2, 0);
  const beta = fit.coef[1]!;
  return { alpha: fit.coef[0]!, beta, seBeta, pValue: 2 * (1 - normalCdf(Math.abs(beta / seBeta))), r2: 1 - fit.ssr / tss, lags: L };
}
