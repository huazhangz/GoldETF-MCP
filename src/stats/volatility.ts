/**
 * Mean + conditional-variance models in the style of Python's `arch` package:
 *
 *   y_t = mu + sum_i phi_i y_{t-i} + e_t,   e_t = sigma_t z_t
 *   variance: constant (sigma_t^2 = sigma^2) or GARCH(1,1) (sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2)
 *   z_t ~ N(0,1) or standardized Student-t(nu)
 *
 * Conventions copied from `arch` so log-likelihoods match exactly at the same parameters:
 * - AR models condition on the first `lags` observations
 * - the GARCH recursion starts from a backcast = EWMA(0.94) of squared OLS residuals (first 75 obs),
 *   computed once from the starting (OLS) mean estimates and held fixed
 * - robust (Bollerslev-Wooldridge sandwich) standard errors by default
 *
 * Internally the data are rescaled to unit variance for numerical stability; results are reported
 * on the original scale (the model is scale-equivariant).
 */
import { normalLogPdf, normalCdf, stdTLogPdf, studentTQuantile, normalQuantile } from "./distributions.js";
import { invert } from "./linalg.js";
import { minimize, numericHessian, numericJacobian } from "./optimize.js";
import { ols } from "./ols.js";

export type MeanModel = "zero" | "constant" | "ar";
export type VarianceModel = "constant" | "garch";
export type Distribution = "normal" | "t";

export interface VolSpec {
  mean: MeanModel;
  /** AR lags when mean = "ar". */
  lags: number;
  variance: VarianceModel;
  dist: Distribution;
}

export interface VolParams {
  mu: number; // 0 when mean = "zero"
  phi: number[];
  /** sigma^2 for constant variance, omega for GARCH. */
  omega: number;
  alpha: number;
  beta: number;
  nu: number; // Infinity for normal
}

export interface VolFit {
  spec: VolSpec;
  params: VolParams;
  coefficients: { name: string; estimate: number; stdErr: number | null; pValue: number | null }[];
  loglik: number;
  nobs: number;
  k: number;
  aic: number;
  bic: number;
  residuals: number[];
  conditionalVariance: number[];
  standardizedResiduals: number[];
  converged: boolean;
  warnings: string[];
  /** GARCH only. */
  persistence?: number;
  halfLife?: number;
  unconditionalVariance?: number;
  /** Internal state for forecasting. */
  state: { y: number[]; lastResid: number; lastSigma2: number };
}

interface Prepared {
  ys: number[]; // dependent (scaled), length nobs
  X: number[][]; // regressors (scaled y lags + constant)
  backcast: number; // scaled
  scale: number;
  olsCoef: number[];
}

const NU_LO = 2.05;
const NU_HI = 500;
const logistic = (u: number) => 1 / (1 + Math.exp(-u));
const logit = (p: number) => Math.log(p / (1 - p));

function prepare(y: number[], spec: VolSpec): Prepared {
  const L = spec.mean === "ar" ? spec.lags : 0;
  const dep: number[] = [];
  const X: number[][] = [];
  for (let t = L; t < y.length; t++) {
    dep.push(y[t]!);
    const row: number[] = [];
    if (spec.mean !== "zero") row.push(1);
    for (let i = 1; i <= L; i++) row.push(y[t - i]!);
    X.push(row);
  }
  let olsCoef: number[] = [];
  let resid = dep;
  if (spec.mean !== "zero") {
    const f = ols(dep, X);
    olsCoef = f.coef;
    resid = dep.map((v, t) => v - X[t]!.reduce((s, x, j) => s + x * f.coef[j]!, 0));
  }
  const scale = Math.sqrt(resid.reduce((s, e) => s + e * e, 0) / resid.length) || 1;
  const tau = Math.min(75, resid.length);
  let wsum = 0;
  let bc = 0;
  for (let i = 0; i < tau; i++) {
    const w = 0.94 ** i;
    wsum += w;
    bc += w * resid[i]! * resid[i]!;
  }
  // Scale: dependent and lagged-y regressors by 1/scale (constant column untouched).
  const ys = dep.map((v) => v / scale);
  const Xs = X.map((row) => row.map((v, j) => (spec.mean !== "zero" && j === 0 ? v : v / scale)));
  const olsScaled = olsCoef.map((c, j) => (j === 0 ? c / scale : c));
  return { ys, X: Xs, backcast: bc / wsum / (scale * scale), scale, olsCoef: olsScaled };
}

/** Natural (scaled) parameter vector layout: [mean coefs..., omega|sigma2, alpha?, beta?, nu?]. */
function layout(spec: VolSpec) {
  const nMean = (spec.mean === "zero" ? 0 : 1) + (spec.mean === "ar" ? spec.lags : 0);
  const nVar = spec.variance === "garch" ? 3 : 1;
  return { nMean, nVar, k: nMean + nVar + (spec.dist === "t" ? 1 : 0) };
}

/** Per-observation log-likelihood contributions at natural (scaled) parameters. */
function contributions(prep: Prepared, spec: VolSpec, theta: number[]): { ll: number[]; e: number[]; s2: number[] } | null {
  const { nMean } = layout(spec);
  const meanC = theta.slice(0, nMean);
  const omega = theta[nMean]!;
  const alpha = spec.variance === "garch" ? theta[nMean + 1]! : 0;
  const beta = spec.variance === "garch" ? theta[nMean + 2]! : 0;
  const nu = spec.dist === "t" ? theta.at(-1)! : Infinity;
  if (!(omega > 0) || alpha < 0 || beta < 0 || alpha + beta >= 1 || (spec.dist === "t" && !(nu > 2))) return null;
  const n = prep.ys.length;
  const e = new Array<number>(n);
  const s2 = new Array<number>(n);
  const ll = new Array<number>(n);
  for (let t = 0; t < n; t++) {
    let fit = 0;
    const row = prep.X[t]!;
    for (let j = 0; j < nMean; j++) fit += row[j]! * meanC[j]!;
    e[t] = prep.ys[t]! - fit;
    if (spec.variance === "garch") {
      s2[t] = t === 0 ? omega + (alpha + beta) * prep.backcast : omega + alpha * e[t - 1]! * e[t - 1]! + beta * s2[t - 1]!;
    } else {
      s2[t] = omega;
    }
    ll[t] = spec.dist === "t" ? stdTLogPdf(e[t]!, s2[t]!, nu) : normalLogPdf(e[t]!, s2[t]!);
  }
  return { ll, e, s2 };
}

function toTheta(spec: VolSpec, u: number[]): number[] {
  const { nMean } = layout(spec);
  const out = u.slice(0, nMean);
  let i = nMean;
  if (spec.variance === "garch") {
    const omega = Math.exp(u[i++]!);
    const pers = logistic(u[i++]!);
    const share = logistic(u[i++]!);
    out.push(omega, pers * share, pers * (1 - share));
  } else {
    out.push(Math.exp(u[i++]!));
  }
  if (spec.dist === "t") out.push(NU_LO + (NU_HI - NU_LO) * logistic(u[i++]!));
  return out;
}

function fromTheta(spec: VolSpec, theta: number[]): number[] {
  const { nMean } = layout(spec);
  const out = theta.slice(0, nMean);
  let i = nMean;
  const clampP = (p: number) => Math.min(1 - 1e-9, Math.max(1e-9, p));
  if (spec.variance === "garch") {
    const omega = theta[i++]!;
    const a = theta[i++]!;
    const b = theta[i++]!;
    const pers = clampP(a + b);
    out.push(Math.log(omega), logit(pers), logit(clampP(a / Math.max(a + b, 1e-12))));
  } else {
    out.push(Math.log(theta[i++]!));
  }
  if (spec.dist === "t") out.push(logit(clampP((theta[i++]! - NU_LO) / (NU_HI - NU_LO))));
  return out;
}

export interface VolFitOptions {
  start?: VolParams;
  robust?: boolean;
  computeStdErr?: boolean;
  fast?: boolean;
}

export function fitVolModel(y: number[], spec: VolSpec, opts: VolFitOptions = {}): VolFit {
  const { nMean, k } = layout(spec);
  const prep = prepare(y, spec);
  const n = prep.ys.length;
  if (n < k + 10) throw new Error(`Too few observations (${n}) for this model (${k} parameters).`);

  const objective = (u: number[]) => {
    const c = contributions(prep, spec, toTheta(spec, u));
    if (!c) return Infinity;
    let s = 0;
    for (const v of c.ll) s += v;
    return -s / n;
  };

  const starts: number[][] = [];
  if (opts.start) {
    const s = opts.start;
    const sc = prep.scale;
    const theta = [
      ...(spec.mean === "zero" ? [] : [s.mu / sc]),
      ...(spec.mean === "ar" ? s.phi : []),
      s.omega / (sc * sc),
      ...(spec.variance === "garch" ? [s.alpha, s.beta] : []),
      ...(spec.dist === "t" ? [s.nu] : []),
    ];
    starts.push(fromTheta(spec, theta));
  } else {
    // Include a near-boundary start (alpha ~ 0, beta high): GARCH likelihoods are often flat and multimodal.
    const grid = spec.variance === "garch" ? [[0.05, 0.9], [0.1, 0.8], [0.15, 0.6], [0.03, 0.5], [0.002, 0.95]] : [[0, 0]];
    for (const nuStart of spec.dist === "t" ? [8, 4] : [Infinity]) {
      for (const [a, b] of grid) {
        const theta = [
          ...prep.olsCoef,
          spec.variance === "garch" ? 1 - a! - b! : 1,
          ...(spec.variance === "garch" ? [a!, b!] : []),
          ...(spec.dist === "t" ? [nuStart] : []),
        ];
        starts.push(fromTheta(spec, theta));
      }
    }
  }
  // Optimize from the three most promising starts and keep the best optimum.
  const ranked = starts.map((s) => ({ s, v: objective(s) })).sort((a, b) => a.v - b.v).slice(0, opts.start ? 1 : 3);
  let best = { x: starts[0]!, fx: Infinity, converged: false };
  for (const { s } of ranked) {
    const r = minimize(objective, s, { step: opts.start ? 0.05 : 0.3, restarts: opts.start ? 1 : 3, ftol: opts.fast ? 1e-9 : 1e-12 });
    if (r.fx < best.fx) best = r;
  }
  if (!Number.isFinite(best.fx)) throw new Error("Volatility model likelihood could not be evaluated.");

  const theta = toTheta(spec, best.x);
  const c = contributions(prep, spec, theta)!;
  const sc = prep.scale;
  const llScaled = c.ll.reduce((s, v) => s + v, 0);
  const loglik = llScaled - n * Math.log(sc);

  // Map scaled parameters back: constant * sc, phi unchanged, omega * sc^2.
  const unscale = theta.map((v, i) => {
    if (spec.mean !== "zero" && i === 0) return v * sc;
    if (i === nMean) return v * sc * sc;
    return v;
  });
  const jac = theta.map((_, i) => (spec.mean !== "zero" && i === 0 ? sc : i === nMean ? sc * sc : 1));

  let se: (number | null)[] = theta.map(() => null);
  const warnings: string[] = [];
  if (opts.computeStdErr ?? true) {
    // Parameters on a boundary (alpha ~ 0, and then beta is unidentified) are held fixed; standard
    // errors for the rest are conditional on them.
    const alphaAtZero = spec.variance === "garch" && theta[nMean + 1]! < 1e-4;
    const free = theta.map((_, i) => i).filter((i) => !(alphaAtZero && (i === nMean + 1 || i === nMean + 2)));
    const full = (sub: number[]) => {
      const th = [...theta];
      free.forEach((idx, j) => (th[idx] = sub[j]!));
      return th;
    };
    const sub0 = free.map((i) => theta[i]!);
    try {
      const negll = (sub: number[]) => {
        const r = contributions(prep, spec, full(sub));
        return r ? -r.ll.reduce((s, v) => s + v, 0) : NaN;
      };
      const H = numericHessian(negll, sub0).map((row) => row.map((v) => v / n));
      const Hinv = invert(H);
      let cov: number[][];
      if (opts.robust ?? true) {
        const scores = numericJacobian((sub) => {
          const r = contributions(prep, spec, full(sub));
          return r ? r.ll.map((v) => -v) : new Array<number>(n).fill(NaN);
        }, sub0);
        const k2 = sub0.length;
        const mean = sub0.map((_, j) => scores.reduce((s, row) => s + row[j]!, 0) / n);
        const S = Array.from({ length: k2 }, (_, a) =>
          Array.from({ length: k2 }, (_, b) => scores.reduce((s, row) => s + (row[a]! - mean[a]!) * (row[b]! - mean[b]!), 0) / (n - 1)),
        );
        const HS = Hinv.map((row) => sub0.map((_, j) => row.reduce((s, v, l) => s + v * S[l]![j]!, 0)));
        cov = HS.map((row) => sub0.map((_, j) => row.reduce((s, v, l) => s + v * Hinv[l]![j]!, 0) / n));
      } else {
        cov = Hinv.map((row) => row.map((v) => v / n));
      }
      free.forEach((idx, j) => {
        const v = cov[j]![j]!;
        se[idx] = v > 0 && Number.isFinite(v) ? Math.sqrt(v) * jac[idx]! : null;
      });
      if (free.some((idx) => se[idx] === null)) warnings.push("Some standard errors are unavailable (information matrix not positive definite).");
      if (alphaAtZero) warnings.push("Standard errors for alpha and beta are not reported because alpha is on its lower bound; the others are conditional on them.");
    } catch {
      warnings.push("Standard errors unavailable (information matrix is singular).");
    }
  }

  const names = [
    ...(spec.mean === "zero" ? [] : ["mu"]),
    ...(spec.mean === "ar" ? Array.from({ length: spec.lags }, (_, i) => `ar${i + 1}`) : []),
    spec.variance === "garch" ? "omega" : "sigma2",
    ...(spec.variance === "garch" ? ["alpha", "beta"] : []),
    ...(spec.dist === "t" ? ["nu"] : []),
  ];
  const coefficients = names.map((name, i) => {
    const s = se[i] ?? null;
    const isTest = !["omega", "sigma2", "nu"].includes(name);
    return { name, estimate: unscale[i]!, stdErr: s, pValue: s && isTest ? 2 * (1 - normalCdf(Math.abs(unscale[i]! / s))) : null };
  });

  const params: VolParams = {
    mu: spec.mean === "zero" ? 0 : unscale[0]!,
    phi: spec.mean === "ar" ? unscale.slice(1, 1 + spec.lags) : [],
    omega: unscale[nMean]!,
    alpha: spec.variance === "garch" ? unscale[nMean + 1]! : 0,
    beta: spec.variance === "garch" ? unscale[nMean + 2]! : 0,
    nu: spec.dist === "t" ? unscale.at(-1)! : Infinity,
  };

  const out: VolFit = {
    spec,
    params,
    coefficients,
    loglik,
    nobs: n,
    k,
    aic: -2 * loglik + 2 * k,
    bic: -2 * loglik + k * Math.log(n),
    residuals: c.e.map((v) => v * sc),
    conditionalVariance: c.s2.map((v) => v * sc * sc),
    standardizedResiduals: c.e.map((v, t) => v / Math.sqrt(c.s2[t]!)),
    converged: best.converged,
    warnings,
    state: { y, lastResid: c.e.at(-1)! * sc, lastSigma2: c.s2.at(-1)! * sc * sc },
  };
  if (spec.variance === "garch") {
    const pers = params.alpha + params.beta;
    out.persistence = pers;
    out.halfLife = pers > 0 ? Math.log(0.5) / Math.log(pers) : 0;
    out.unconditionalVariance = params.omega / (1 - pers);
    if (params.alpha < 1e-4) {
      warnings.push("alpha is at its lower bound (~0): no detectable ARCH effect, so beta is not identified. GARCH is not warranted.");
    }
    if (pers > 0.995) warnings.push(`Persistence alpha + beta = ${pers.toFixed(4)} is close to 1 (near-integrated variance).`);
  }
  if (spec.dist === "t" && params.nu > 100) warnings.push("nu > 100: tails are effectively normal; a normal distribution is adequate.");
  if (!best.converged) warnings.push("Optimizer stopped before converging; treat estimates with caution.");
  return out;
}

/** Log-likelihood at given (original-scale) parameters. Used for parity tests against `arch`. */
export function volLoglikAt(y: number[], spec: VolSpec, p: VolParams): number {
  const prep = prepare(y, spec);
  const sc = prep.scale;
  const theta = [
    ...(spec.mean === "zero" ? [] : [p.mu / sc]),
    ...(spec.mean === "ar" ? p.phi : []),
    p.omega / (sc * sc),
    ...(spec.variance === "garch" ? [p.alpha, p.beta] : []),
    ...(spec.dist === "t" ? [p.nu] : []),
  ];
  const c = contributions(prep, spec, theta);
  if (!c) return NaN;
  return c.ll.reduce((s, v) => s + v, 0) - prep.ys.length * Math.log(sc);
}

export interface VolForecast {
  mean: number[];
  /** Standard deviation of the h-step forecast error of y (includes AR propagation). */
  sd: number[];
  /** Conditional variance of the innovation e_{T+h}. */
  residualVariance: number[];
}

export function forecastVol(fit: VolFit, h: number): VolForecast {
  const { params: p, spec } = fit;
  const y = fit.state.y;
  const hist = [...y];
  const mean: number[] = [];
  for (let i = 0; i < h; i++) {
    let m = spec.mean === "zero" ? 0 : p.mu;
    for (let j = 0; j < p.phi.length; j++) m += p.phi[j]! * hist[hist.length - 1 - j]!;
    hist.push(m);
    mean.push(m);
  }
  const rv: number[] = [];
  for (let i = 0; i < h; i++) {
    if (spec.variance === "garch") {
      rv.push(i === 0 ? p.omega + p.alpha * fit.state.lastResid ** 2 + p.beta * fit.state.lastSigma2 : p.omega + (p.alpha + p.beta) * rv[i - 1]!);
    } else rv.push(p.omega);
  }
  // psi-weights of the AR polynomial
  const psi = [1];
  for (let j = 1; j < h; j++) {
    let v = 0;
    for (let i = 1; i <= Math.min(j, p.phi.length); i++) v += p.phi[i - 1]! * psi[j - i]!;
    psi.push(v);
  }
  const sd = rv.map((_, step) => {
    let v = 0;
    for (let j = 0; j <= step; j++) v += psi[j]! ** 2 * rv[step - j]!;
    return Math.sqrt(v);
  });
  return { mean, sd, residualVariance: rv };
}

/** Two-sided central interval half-width multiplier for the model's innovation distribution. */
export function intervalMultiplier(level: number, nu: number): number {
  const p = 0.5 + level / 2;
  if (!Number.isFinite(nu)) return normalQuantile(p);
  return studentTQuantile(p, nu) * Math.sqrt((nu - 2) / nu);
}
