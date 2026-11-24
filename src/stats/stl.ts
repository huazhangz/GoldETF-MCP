/**
 * STL: Seasonal-Trend decomposition using Loess (Cleveland, Cleveland, McRae & Terpenning 1990).
 * A direct port of the reference Fortran (as used by R's stats::stl and statsmodels' STL) with
 * all "jump" parameters fixed at 1 (no interpolation shortcuts).
 */

export interface StlOptions {
  period: number;
  /** Seasonal smoother span (odd, >= 3), or "periodic" (R's s.window = "periodic"). Default "periodic". */
  seasonal?: number | "periodic";
  trend?: number;
  lowPass?: number;
  robust?: boolean;
  /** Override iteration counts (defaults follow R: robust -> inner 1, outer 15; else inner 2, outer 0). */
  inner?: number;
  outer?: number;
}

export interface StlResult {
  seasonal: number[];
  trend: number[];
  remainder: number[];
  weights: number[];
  /** Strength of trend / seasonality (Wang, Smith & Hyndman 2006): max(0, 1 - Var(R)/Var(X + R)). */
  trendStrength: number;
  seasonalStrength: number;
  params: { period: number; seasonal: number; trend: number; lowPass: number; inner: number; outer: number; seasonalDeg: number };
}

const nextOdd = (x: number) => {
  const v = Math.ceil(x);
  return v % 2 === 0 ? v + 1 : v;
};

/** Loess estimate at xs (1-based positions) using points nleft..nright. Returns null when all weights vanish. */
function est(
  y: number[],
  n: number,
  len: number,
  deg: number,
  xs: number,
  nleft: number,
  nright: number,
  w: number[],
  userw: boolean,
  rw: number[],
): number | null {
  const range = n - 1;
  let h = Math.max(xs - nleft, nright - xs);
  if (len > n) h += Math.trunc((len - n) / 2);
  const h9 = 0.999 * h;
  const h1 = 0.001 * h;
  let a = 0;
  for (let j = nleft; j <= nright; j++) {
    w[j - 1] = 0;
    const r = Math.abs(j - xs);
    if (r <= h9) {
      w[j - 1] = r <= h1 ? 1 : (1 - (r / h) ** 3) ** 3;
      if (userw) w[j - 1]! *= rw[j - 1]!;
      a += w[j - 1]!;
    }
  }
  if (a <= 0) return null;
  for (let j = nleft; j <= nright; j++) w[j - 1]! /= a;
  if (h > 0 && deg > 0) {
    let am = 0;
    for (let j = nleft; j <= nright; j++) am += w[j - 1]! * j;
    let b = xs - am;
    let c = 0;
    for (let j = nleft; j <= nright; j++) c += w[j - 1]! * (j - am) ** 2;
    if (Math.sqrt(c) > 0.001 * range) {
      b /= c;
      for (let j = nleft; j <= nright; j++) w[j - 1]! *= b * (j - am) + 1;
    }
  }
  let ys = 0;
  for (let j = nleft; j <= nright; j++) ys += w[j - 1]! * y[j - 1]!;
  return ys;
}

/** Loess smooth of y (length n) with span len, evaluated at every point. */
function ess(y: number[], n: number, len: number, deg: number, userw: boolean, rw: number[]): number[] {
  const ys = new Array<number>(n);
  if (n < 2) {
    ys[0] = y[0]!;
    return ys;
  }
  const w = new Array<number>(n).fill(0);
  let nleft = 1;
  let nright = Math.min(len, n);
  const nsh = Math.trunc((len + 1) / 2);
  for (let i = 1; i <= n; i++) {
    if (len < n && i > nsh && nright !== n) {
      nleft++;
      nright++;
    }
    const v = est(y, n, len, deg, i, nleft, nright, w, userw, rw);
    ys[i - 1] = v ?? y[i - 1]!;
  }
  return ys;
}

/** Moving average of length len. */
function ma(x: number[], len: number): number[] {
  const out: number[] = [];
  let s = 0;
  for (let i = 0; i < len; i++) s += x[i]!;
  out.push(s / len);
  for (let i = len; i < x.length; i++) {
    s += x[i]! - x[i - len]!;
    out.push(s / len);
  }
  return out;
}

/** Cycle-subseries smoothing; returns an array of length n + 2*np (extrapolated one period each side). */
function ss(y: number[], n: number, np: number, ns: number, deg: number, userw: boolean, rw: number[]): number[] {
  const season = new Array<number>(n + 2 * np).fill(0);
  for (let j = 1; j <= np; j++) {
    const k = Math.trunc((n - j) / np) + 1;
    const sub = Array.from({ length: k }, (_, i) => y[i * np + j - 1]!);
    const subw = Array.from({ length: k }, (_, i) => rw[i * np + j - 1]!);
    const sm = ess(sub, k, ns, deg, userw, subw);
    const w = new Array<number>(k).fill(0);
    const first = est(sub, k, ns, deg, 0, 1, Math.min(ns, k), w, userw, subw) ?? sm[0]!;
    const last = est(sub, k, ns, deg, k + 1, Math.max(1, k - ns + 1), k, w, userw, subw) ?? sm[k - 1]!;
    const full = [first, ...sm, last];
    for (let m = 0; m < k + 2; m++) season[m * np + j - 1] = full[m]!;
  }
  return season;
}

function robustnessWeights(y: number[], fit: number[]): number[] {
  const r = y.map((v, i) => Math.abs(v - fit[i]!));
  const sorted = [...r].sort((a, b) => a - b);
  const n = r.length;
  const mid1 = Math.trunc(n / 2) + 1;
  const mid2 = n - mid1 + 1;
  const cmad = 3 * (sorted[mid1 - 1]! + sorted[mid2 - 1]!);
  const c9 = 0.999 * cmad;
  const c1 = 0.001 * cmad;
  return r.map((ri) => (ri <= c1 ? 1 : ri <= c9 ? (1 - (ri / cmad) ** 2) ** 2 : 0));
}

function variance(x: number[]) {
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  return x.reduce((s, v) => s + (v - m) ** 2, 0) / (x.length - 1);
}

export function stl(y: number[], opts: StlOptions): StlResult {
  const n = y.length;
  const np = opts.period;
  if (np < 2) throw new Error("STL period must be at least 2.");
  if (n < 2 * np) throw new Error(`STL needs at least two full periods (${2 * np} observations); got ${n}.`);
  const periodic = (opts.seasonal ?? "periodic") === "periodic";
  const ns = periodic ? 10 * n + 1 : nextOdd(Math.max(3, opts.seasonal as number));
  const isdeg = periodic ? 0 : 1;
  const nt = opts.trend ?? nextOdd((1.5 * np) / (1 - 1.5 / ns));
  const nl = opts.lowPass ?? nextOdd(np);
  const robust = opts.robust ?? false;
  const ni = opts.inner ?? (robust ? 1 : 2);
  const no = opts.outer ?? (robust ? 15 : 0);

  let rw = new Array<number>(n).fill(1);
  let season = new Array<number>(n).fill(0);
  let trend = new Array<number>(n).fill(0);
  let userw = false;
  for (let k = 0; ; k++) {
    for (let it = 0; it < ni; it++) {
      const detr = y.map((v, i) => v - trend[i]!);
      const c = ss(detr, n, np, ns, isdeg, userw, rw);
      const low = ess(ma(ma(ma(c, np), np), 3), n, nl, 1, false, rw);
      season = low.map((l, i) => c[np + i]! - l);
      const deseas = y.map((v, i) => v - season[i]!);
      trend = ess(deseas, n, nt, 1, userw, rw);
    }
    if (k >= no) break;
    rw = robustnessWeights(
      y,
      y.map((_, i) => trend[i]! + season[i]!),
    );
    userw = true;
  }
  if (no <= 0) rw = new Array<number>(n).fill(1);

  if (periodic) {
    // As R does for s.window = "periodic": average the seasonal component by cycle position.
    const sums = new Array<number>(np).fill(0);
    const counts = new Array<number>(np).fill(0);
    season.forEach((s, i) => {
      sums[i % np]! += s;
      counts[i % np]!++;
    });
    season = season.map((_, i) => sums[i % np]! / counts[i % np]!);
  }
  const remainder = y.map((v, i) => v - season[i]! - trend[i]!);
  const vr = variance(remainder);
  return {
    seasonal: season,
    trend,
    remainder,
    weights: rw,
    trendStrength: Math.max(0, 1 - vr / variance(trend.map((t, i) => t + remainder[i]!))),
    seasonalStrength: Math.max(0, 1 - vr / variance(season.map((s, i) => s + remainder[i]!))),
    params: { period: np, seasonal: ns, trend: nt, lowPass: nl, inner: ni, outer: no, seasonalDeg: isdeg },
  };
}
