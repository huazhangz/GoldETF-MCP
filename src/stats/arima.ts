/**
 * ARIMA(p, d, q) with an optional constant, fitted by exact Gaussian maximum likelihood.
 *
 * The series is differenced d times and an ARMA(p, q) with mean mu is fitted to the result
 * with a Kalman filter (Harvey state-space form, stationary initialization). For d = 1, mu is
 * the drift; for d = 0 it is the process mean. This matches statsmodels' ARIMA(..., trend="c"/"t")
 * and R's forecast::Arima(include.mean / include.drift).
 *
 * Stationarity and invertibility are enforced by optimizing over partial autocorrelations
 * (tanh transform + Durbin-Levinson), the same idea as statsmodels' Monahan/Jones transform.
 */
import { normalCdf } from "./distributions.js";
import { invert, polyRoots, SingularMatrixError, solve } from "./linalg.js";
import { minimize, numericHessian } from "./optimize.js";
import { ols } from "./ols.js";

export interface ArimaSpec {
  p: number;
  d: number;
  q: number;
  /** Include a mean (d = 0) or drift (d = 1). Ignored (forced false) when d >= 2. */
  constant: boolean;
}

export interface ArimaParams {
  mu: number;
  ar: number[];
  ma: number[];
  sigma2: number;
}

export interface Coefficient {
  name: string;
  estimate: number;
  stdErr: number | null;
  pValue: number | null;
}

export interface ArimaFit {
  spec: ArimaSpec;
  params: ArimaParams;
  coefficients: Coefficient[];
  loglik: number;
  nobs: number;
  /** Number of estimated parameters, including sigma2. */
  k: number;
  aic: number;
  aicc: number;
  bic: number;
  /** One-step-ahead innovations of the differenced series (length n - d). */
  residuals: number[];
  /** Innovations divided by their predicted standard deviation. */
  standardizedResiduals: number[];
  converged: boolean;
  warnings: string[];
  /** Inverse roots (modulus) of the AR and MA polynomials; all < 1 by construction. */
  arInverseRoots: [number, number][];
  maInverseRoots: [number, number][];
  /** Internal: what forecasting needs. */
  state: { y: number[]; a: number[]; P: number[][] };
}

/* ----------------------------- transforms ----------------------------- */

export function pacfToAr(r: number[]): number[] {
  let phi: number[] = [];
  for (let k = 0; k < r.length; k++) {
    const rk = r[k]!;
    const next = phi.map((p, j) => p - rk * phi[k - 1 - j]!);
    next.push(rk);
    phi = next;
  }
  return phi;
}

export function arToPacf(phi: number[]): number[] | null {
  let cur = [...phi];
  const out = new Array<number>(phi.length);
  for (let k = phi.length; k >= 1; k--) {
    const rk = cur[k - 1]!;
    if (!(Math.abs(rk) < 1)) return null;
    out[k - 1] = rk;
    const prev = new Array<number>(k - 1);
    for (let j = 0; j < k - 1; j++) prev[j] = (cur[j]! + rk * cur[k - 2 - j]!) / (1 - rk * rk);
    cur = prev;
  }
  return out;
}

const constrain = (u: number[]) => pacfToAr(u.map(Math.tanh));
const unconstrain = (phi: number[]) => arToPacf(phi)?.map((r) => Math.atanh(Math.max(-0.999, Math.min(0.999, r)))) ?? null;

/* ----------------------------- state space ----------------------------- */

interface Model {
  r: number;
  ar: number[]; // padded to length r
  R: number[];
}

function makeModel(ar: number[], ma: number[]): Model {
  const r = Math.max(ar.length, ma.length + 1);
  const arP = Array.from({ length: r }, (_, i) => ar[i] ?? 0);
  const R = Array.from({ length: r }, (_, i) => (i === 0 ? 1 : (ma[i - 1] ?? 0)));
  return { r, ar: arP, R };
}

/** T x for the companion transition matrix. */
function tVec(m: Model, x: number[]): number[] {
  return x.map((_, i) => m.ar[i]! * x[0]! + (i + 1 < m.r ? x[i + 1]! : 0));
}

/** T M T' + R R'. */
function tMtPlusQ(m: Model, M: number[][]): number[][] {
  const r = m.r;
  const A = M.map((_, i) => M[0]!.map((_, j) => m.ar[i]! * M[0]![j]! + (i + 1 < r ? M[i + 1]![j]! : 0))); // T M
  return A.map((row, i) =>
    row.map((_, j) => m.ar[j]! * row[0]! + (j + 1 < r ? row[j + 1]! : 0) + m.R[i]! * m.R[j]!),
  );
}

/** Unconditional state covariance (unit innovation variance): P = T P T' + R R'. */
function initialCov(m: Model): number[][] {
  const r = m.r;
  const n = r * r;
  const T = (i: number, k: number) => (k === 0 ? m.ar[i]! : 0) + (k === i + 1 ? 1 : 0);
  const A = Array.from({ length: n }, (_, row) => {
    const i = Math.floor(row / r);
    const j = row % r;
    return Array.from({ length: n }, (_, col) => {
      const k = Math.floor(col / r);
      const l = col % r;
      return (row === col ? 1 : 0) - T(i, k) * T(j, l);
    });
  });
  const b = Array.from({ length: n }, (_, row) => [m.R[Math.floor(row / r)]! * m.R[row % r]!]);
  const v = solve(A, b);
  return Array.from({ length: r }, (_, i) => Array.from({ length: r }, (_, j) => v[i * r + j]![0]!));
}

interface FilterOut {
  v: number[];
  F: number[];
  a: number[];
  P: number[][];
}

/**
 * Kalman filter for the ARMA state-space model. Written with flat typed arrays because it is the
 * hot loop of every fit (it runs thousands of times per optimization).
 */
function kalman(w: number[], mu: number, m: Model): FilterOut | null {
  const r = m.r;
  let P0: number[][];
  try {
    P0 = initialCov(m);
  } catch (e) {
    if (e instanceof SingularMatrixError) return null;
    throw e;
  }
  const ar = Float64Array.from(m.ar);
  const R = Float64Array.from(m.R);
  const P = new Float64Array(r * r);
  for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) P[i * r + j] = P0[i]![j]!;
  const A = new Float64Array(r * r);
  const Pn = new Float64Array(r * r);
  const Pz = new Float64Array(r);
  const a = new Float64Array(r);
  const att = new Float64Array(r);
  const n = w.length;
  const v = new Array<number>(n);
  const F = new Array<number>(n);
  let steady = false;
  for (let t = 0; t < n; t++) {
    const vt = w[t]! - mu - a[0]!;
    const Ft = P[0]!;
    if (!(Ft > 0)) return null;
    v[t] = vt;
    F[t] = Ft;
    for (let i = 0; i < r; i++) Pz[i] = P[i * r]!;
    const k = vt / Ft;
    for (let i = 0; i < r; i++) att[i] = a[i]! + Pz[i]! * k;
    // a = T att
    for (let i = 0; i < r; i++) a[i] = ar[i]! * att[0]! + (i + 1 < r ? att[i + 1]! : 0);
    if (!steady) {
      // Ptt = P - Pz Pz' / F  (stored in Pn), then A = T Ptt, Pn = A T' + R R'
      for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) Pn[i * r + j] = P[i * r + j]! - (Pz[i]! * Pz[j]!) / Ft;
      for (let i = 0; i < r; i++)
        for (let j = 0; j < r; j++) A[i * r + j] = ar[i]! * Pn[j]! + (i + 1 < r ? Pn[(i + 1) * r + j]! : 0);
      let diff = 0;
      for (let i = 0; i < r; i++)
        for (let j = 0; j < r; j++) {
          const val = ar[j]! * A[i * r]! + (j + 1 < r ? A[i * r + j + 1]! : 0) + R[i]! * R[j]!;
          const dlt = Math.abs(val - P[i * r + j]!);
          if (dlt > diff) diff = dlt;
          Pn[i * r + j] = val;
        }
      P.set(Pn);
      if (diff < 1e-15) steady = true;
    }
  }
  const Pout = Array.from({ length: r }, (_, i) => Array.from({ length: r }, (_, j) => P[i * r + j]!));
  return { v, F, a: Array.from(a), P: Pout };
}

function concentratedLoglik(f: FilterOut): { ll: number; sigma2: number } {
  const n = f.v.length;
  let s = 0;
  let lf = 0;
  for (let t = 0; t < n; t++) {
    s += (f.v[t]! * f.v[t]!) / f.F[t]!;
    lf += Math.log(f.F[t]!);
  }
  const sigma2 = s / n;
  return { ll: -0.5 * n * (Math.log(2 * Math.PI) + 1 + Math.log(sigma2)) - 0.5 * lf, sigma2 };
}

function fullLoglik(f: FilterOut, sigma2: number): number {
  let ll = 0;
  for (let t = 0; t < f.v.length; t++) {
    const s = sigma2 * f.F[t]!;
    ll -= 0.5 * (Math.log(2 * Math.PI) + Math.log(s) + (f.v[t]! * f.v[t]!) / s);
  }
  return ll;
}

/* ----------------------------- fitting ----------------------------- */

export function difference(y: number[], d: number): number[] {
  let w = y;
  for (let i = 0; i < d; i++) w = w.slice(1).map((v, j) => v - w[j]!);
  return w;
}

function meanOf(x: number[]) {
  return x.reduce((s, v) => s + v, 0) / x.length;
}

/** Hannan-Rissanen starting values; null if they violate stationarity/invertibility. */
function hannanRissanen(x: number[], p: number, q: number): { ar: number[]; ma: number[] } | null {
  const n = x.length;
  if (q === 0 && p === 0) return { ar: [], ma: [] };
  const m = Math.min(Math.max(p, q) + 6, Math.floor(n / 4));
  let e = new Array<number>(n).fill(0);
  if (q > 0) {
    const rows: number[][] = [];
    const ys: number[] = [];
    for (let t = m; t < n; t++) {
      rows.push(Array.from({ length: m }, (_, i) => x[t - 1 - i]!));
      ys.push(x[t]!);
    }
    try {
      const f = ols(ys, rows);
      e = x.map((v, t) => (t < m ? 0 : v - rows[t - m]!.reduce((s, r, i) => s + r * f.coef[i]!, 0)));
    } catch {
      return null;
    }
  }
  const start = Math.max(p, q) + (q > 0 ? m : 0);
  const rows: number[][] = [];
  const ys: number[] = [];
  for (let t = start; t < n; t++) {
    rows.push([...Array.from({ length: p }, (_, i) => x[t - 1 - i]!), ...Array.from({ length: q }, (_, i) => e[t - 1 - i]!)]);
    ys.push(x[t]!);
  }
  if (rows.length <= p + q + 2) return null;
  try {
    const f = ols(ys, rows);
    const ar = f.coef.slice(0, p);
    const ma = f.coef.slice(p);
    if (!unconstrain(ar) || !unconstrain(ma.map((v) => -v))) return null;
    return { ar, ma };
  } catch {
    return null;
  }
}

export interface FitOptions {
  /** Warm start (e.g. the previous fit in a backtest). */
  start?: ArimaParams;
  computeStdErr?: boolean;
  /** Looser optimizer tolerance; adequate for forecasting (used by backtests). */
  fast?: boolean;
}

export function fitArima(y: number[], spec: ArimaSpec, opts: FitOptions = {}): ArimaFit {
  const { p, d, q } = spec;
  const constant = spec.constant && d < 2;
  const w = difference(y, d);
  const n = w.length;
  const k = (constant ? 1 : 0) + p + q + 1;
  if (n < k + 5) {
    throw new Error(`Too few observations (${n} after differencing) for ARIMA(${p},${d},${q}).`);
  }
  const wMean = meanOf(w);
  const wSd = Math.sqrt(w.reduce((s, v) => s + (v - wMean) ** 2, 0) / n) || 1;

  const unpack = (u: number[]) => {
    let i = 0;
    const mu = constant ? wMean + wSd * u[i++]! : 0;
    const ar = constrain(u.slice(i, i + p));
    i += p;
    const ma = constrain(u.slice(i, i + q)).map((v) => -v);
    return { mu, ar, ma };
  };
  const pack = (mu: number, ar: number[], ma: number[]): number[] | null => {
    const ua = unconstrain(ar);
    const um = unconstrain(ma.map((v) => -v));
    if (!ua || !um) return null;
    return [...(constant ? [(mu - wMean) / wSd] : []), ...ua, ...um];
  };
  const objective = (u: number[]) => {
    const { mu, ar, ma } = unpack(u);
    const f = kalman(w, mu, makeModel(ar, ma));
    return f ? -concentratedLoglik(f).ll / n : Infinity;
  };

  const starts: number[][] = [];
  if (opts.start) {
    const s = pack(opts.start.mu, opts.start.ar, opts.start.ma);
    if (s) starts.push(s);
  }
  if (starts.length === 0) {
    starts.push(new Array<number>((constant ? 1 : 0) + p + q).fill(0));
    const hr = hannanRissanen(w.map((v) => v - (constant ? wMean : 0)), p, q);
    if (hr && (p > 0 || q > 0)) {
      const s = pack(wMean, hr.ar, hr.ma);
      if (s) starts.push(s);
    }
  }
  let best = { x: starts[0]!, fx: Infinity, converged: false };
  for (const s of starts) {
    const r = minimize(objective, s, {
      step: opts.start ? 0.05 : 0.3,
      restarts: opts.start ? 1 : 2,
      ftol: opts.fast ? 1e-9 : 1e-12,
    });
    if (r.fx < best.fx) best = r;
  }
  if (!Number.isFinite(best.fx)) throw new Error(`ARIMA(${p},${d},${q}) likelihood could not be evaluated.`);

  const { mu, ar, ma } = unpack(best.x);
  const model = makeModel(ar, ma);
  const filt = kalman(w, mu, model)!;
  const { ll, sigma2 } = concentratedLoglik(filt);
  const params: ArimaParams = { mu, ar, ma, sigma2 };
  return buildFit(y, spec, constant, params, filt, ll, best.converged, opts.computeStdErr ?? true);
}

/** Evaluate an ARIMA at given parameters (no optimization). Used for parity tests and forecasting. */
export function arimaAtParams(y: number[], spec: ArimaSpec, params: ArimaParams): ArimaFit {
  const constant = spec.constant && spec.d < 2;
  const w = difference(y, spec.d);
  const filt = kalman(w, constant ? params.mu : 0, makeModel(params.ar, params.ma));
  if (!filt) throw new Error("Parameters imply a non-stationary or degenerate model.");
  return buildFit(y, spec, constant, params, filt, fullLoglik(filt, params.sigma2), true, true);
}

function buildFit(
  y: number[],
  spec: ArimaSpec,
  constant: boolean,
  params: ArimaParams,
  filt: FilterOut,
  ll: number,
  converged: boolean,
  computeStdErr: boolean,
): ArimaFit {
  const { p, d, q } = spec;
  const w = difference(y, d);
  const n = w.length;
  const k = (constant ? 1 : 0) + p + q + 1;
  const names = [
    ...(constant ? [d === 0 ? "mean" : "drift"] : []),
    ...params.ar.map((_, i) => `ar${i + 1}`),
    ...params.ma.map((_, i) => `ma${i + 1}`),
    "sigma2",
  ];
  const est = [...(constant ? [params.mu] : []), ...params.ar, ...params.ma, params.sigma2];

  let se: (number | null)[] = est.map(() => null);
  const warnings: string[] = [];
  if (computeStdErr) {
    const negll = (x: number[]) => {
      let i = 0;
      const mu = constant ? x[i++]! : 0;
      const ar = x.slice(i, i + p);
      i += p;
      const ma = x.slice(i, i + q);
      const s2 = x[k - 1]!;
      if (!(s2 > 0)) return NaN;
      const f = kalman(w, mu, makeModel(ar, ma));
      return f ? -fullLoglik(f, s2) : NaN;
    };
    try {
      const H = numericHessian(negll, est);
      if (H.every((row) => row.every(Number.isFinite))) {
        const cov = invert(H);
        se = cov.map((row, i) => (row[i]! > 0 ? Math.sqrt(row[i]!) : null));
      }
    } catch {
      /* leave null */
    }
    if (se.some((s) => s === null)) {
      warnings.push("Standard errors unavailable (information matrix not positive definite): the model is likely over-parameterized.");
    }
  }

  const coefficients: Coefficient[] = names.map((name, i) => {
    const s = se[i] ?? null;
    return {
      name,
      estimate: est[i]!,
      stdErr: s,
      pValue: s && name !== "sigma2" ? 2 * (1 - normalCdf(Math.abs(est[i]! / s))) : null,
    };
  });

  const arInverseRoots = inverseRoots([1, ...params.ar.map((v) => -v)]);
  const maInverseRoots = inverseRoots([1, ...params.ma]);
  for (const [i, c] of coefficients.entries()) {
    if (i < k - 1 && c.pValue !== null && c.pValue > 0.05) {
      warnings.push(`${c.name} is not significant (p = ${c.pValue.toFixed(3)}).`);
    }
  }
  let closest = Infinity;
  for (const a of arInverseRoots) for (const m of maInverseRoots) closest = Math.min(closest, Math.hypot(a[0] - m[0], a[1] - m[1]));
  if (closest < 0.1) {
    warnings.push(
      "An AR root nearly cancels an MA root: the AR and MA terms offset each other, so a model with lower p and q is likely equivalent.",
    );
  }
  const maxAr = Math.max(0, ...arInverseRoots.map(([re, im]) => Math.hypot(re, im)));
  if (d === 0 && maxAr > 0.97) warnings.push(`Near unit root (largest AR inverse root modulus ${maxAr.toFixed(3)}); consider d = 1.`);
  const maxMa = Math.max(0, ...maInverseRoots.map(([re, im]) => Math.hypot(re, im)));
  // An MA root near +1 after differencing nearly cancels the (1 - B) factor: the classic over-differencing signature.
  const unitMa = Math.max(0, ...maInverseRoots.filter(([re, im]) => Math.abs(im) < 1e-6 && re > 0).map(([re]) => re));
  if (d >= 1 && unitMa > 0.9) {
    warnings.push(
      `MA inverse root ${unitMa.toFixed(3)} is close to 1 and nearly cancels the differencing: the series is likely over-differenced (try d = ${d - 1} with an AR term).`,
    );
  } else if (maxMa > 0.97) {
    warnings.push(`Near non-invertible MA (inverse root modulus ${maxMa.toFixed(3)}).`);
  }
  if (!converged) warnings.push("Optimizer stopped before converging; treat estimates with caution.");

  return {
    spec: { ...spec, constant },
    params,
    coefficients,
    loglik: ll,
    nobs: n,
    k,
    aic: -2 * ll + 2 * k,
    aicc: -2 * ll + 2 * k + (2 * k * (k + 1)) / Math.max(1, n - k - 1),
    bic: -2 * ll + k * Math.log(n),
    residuals: filt.v,
    standardizedResiduals: filt.v.map((v, t) => v / Math.sqrt(params.sigma2 * filt.F[t]!)),
    converged,
    warnings,
    arInverseRoots,
    maInverseRoots,
    state: { y, a: filt.a, P: filt.P },
  };
}

function inverseRoots(poly: number[]): [number, number][] {
  return polyRoots(poly).map(([re, im]) => {
    const m2 = re * re + im * im;
    return [re / m2, -im / m2];
  });
}

/* ----------------------------- forecasting ----------------------------- */

export interface Forecast {
  /** Point forecast (conditional mean) on the modeled scale. */
  mean: number[];
  /** Forecast-error standard deviation on the modeled scale. */
  sd: number[];
}

export function forecastArima(fit: ArimaFit, h: number): Forecast {
  const { d } = fit.spec;
  const { mu, ar, ma, sigma2 } = fit.params;
  const m = makeModel(ar, ma);
  const y = fit.state.y;
  const n = y.length;

  // State predictions for steps 1..h (P_i is the predicted state covariance for step i).
  const wHat: number[] = [];
  const Ps: number[][][] = [];
  let a = fit.state.a;
  let P = fit.state.P;
  for (let i = 0; i < h; i++) {
    wHat.push((fit.spec.constant ? mu : 0) + a[0]!);
    Ps.push(P);
    a = tVec(m, a);
    P = tMtPlusQ(m, P);
  }
  // g_k = e1' T^k, used for Cov(err_i, err_j) = sigma2 * g_{j-i}' P_i e1.
  const g: number[][] = [Array.from({ length: m.r }, (_, i) => (i === 0 ? 1 : 0))];
  for (let k = 1; k < h; k++) {
    const prev = g[k - 1]!;
    g.push(prev.map((_, j) => (j === 0 ? prev.reduce((s, gi, i) => s + gi * m.ar[i]!, 0) : prev[j - 1]!)));
  }
  const cov = (i: number, j: number) => {
    const [lo, hi] = i <= j ? [i, j] : [j, i];
    const gk = g[hi - lo]!;
    const Pi = Ps[lo]!;
    return sigma2 * gk.reduce((s, gm, mIdx) => s + gm * Pi[mIdx]![0]!, 0);
  };

  const mean: number[] = [];
  const sd: number[] = [];
  for (let step = 1; step <= h; step++) {
    // y_{n+step} = base + sum_i c_i * w_{n+i}
    const c = Array.from({ length: step }, (_, i) => (d === 0 ? (i === step - 1 ? 1 : 0) : d === 1 ? 1 : step - i));
    const base = d === 0 ? 0 : d === 1 ? y[n - 1]! : y[n - 1]! + step * (y[n - 1]! - y[n - 2]!);
    mean.push(base + c.reduce((s, ci, i) => s + ci * wHat[i]!, 0));
    let v = 0;
    for (let i = 0; i < step; i++) for (let j = 0; j < step; j++) if (c[i] && c[j]) v += c[i]! * c[j]! * cov(i, j);
    sd.push(Math.sqrt(Math.max(v, 0)));
  }
  return { mean, sd };
}
