import { normalCdf } from "./distributions.js";
import { ols } from "./ols.js";

export type Deterministic = "c" | "ct";

export interface AdfResult {
  statistic: number;
  pValue: number;
  usedLag: number;
  nobs: number;
  criticalValues: { "1%": number; "5%": number; "10%": number };
  regression: Deterministic;
  lagSelection: "aic" | "fixed";
}

/*
 * MacKinnon (1994) response-surface coefficients for approximate p-values, N = 1.
 * Same constants as statsmodels.tsa.adfvalues (scaling already applied).
 */
const MACKINNON_P = {
  c: {
    star: -1.61,
    min: -18.83,
    max: 2.74,
    small: [2.1659, 1.4412, 0.038269],
    large: [1.7339, 0.93202, -0.12745, -0.010368],
  },
  ct: {
    star: -2.89,
    min: -16.18,
    max: 0.7,
    small: [3.2512, 1.6047, 0.049588],
    large: [2.5261, 0.61654, -0.37956, -0.060285],
  },
} as const;

/* MacKinnon (2010) finite-sample critical values: b0 + b1/T + b2/T^2 + b3/T^3. */
const MACKINNON_CRIT = {
  c: [
    [-3.43035, -6.5393, -16.786, -79.433],
    [-2.86154, -2.8903, -4.234, -40.04],
    [-2.56677, -1.5384, -2.809, 0],
  ],
  ct: [
    [-3.95877, -9.0531, -28.428, -134.155],
    [-3.41049, -4.3904, -9.036, -45.374],
    [-3.12705, -2.5856, -3.925, -22.38],
  ],
} as const;

function polyval(coef: readonly number[], x: number): number {
  return coef.reduce((s, c, i) => s + c * x ** i, 0);
}

export function mackinnonPValue(stat: number, regression: Deterministic): number {
  const t = MACKINNON_P[regression];
  if (stat > t.max) return 1;
  if (stat < t.min) return 0;
  return normalCdf(polyval(stat <= t.star ? t.small : t.large, stat));
}

function mackinnonCrit(regression: Deterministic, nobs: number): AdfResult["criticalValues"] {
  const [c1, c5, c10] = MACKINNON_CRIT[regression].map((b) => polyval(b, 1 / nobs));
  return { "1%": c1!, "5%": c5!, "10%": c10! };
}

/**
 * Build the ADF regression  Δy_t = [const, trend] + γ y_{t-1} + Σ_{i=1..p} δ_i Δy_{t-i}
 * using observations from index `start` of the differenced series.
 */
function design(x: number[], dx: number[], lags: number, start: number, regression: Deterministic) {
  const y: number[] = [];
  const X: number[][] = [];
  for (let t = start; t < dx.length; t++) {
    const row: number[] = [1];
    if (regression === "ct") row.push(t - start + 1);
    row.push(x[t]!); // level y_{t} corresponds to Δy_{t} = x[t+1] - x[t]
    for (let i = 1; i <= lags; i++) row.push(dx[t - i]!);
    y.push(dx[t]!);
    X.push(row);
  }
  return { y, X, levelIndex: regression === "ct" ? 2 : 1 };
}

export interface AdfOptions {
  regression?: Deterministic;
  /** Fixed number of lagged differences. If omitted, chosen by AIC up to `maxLag`. */
  lags?: number;
  /** Upper bound for AIC search. Default: ceil(12 * (n/100)^(1/4)) (Schwert rule, as statsmodels). */
  maxLag?: number;
}

/** Augmented Dickey-Fuller unit-root test. H0: the series has a unit root (non-stationary). */
export function adfTest(x: number[], opts: AdfOptions = {}): AdfResult {
  const regression = opts.regression ?? "c";
  const n = x.length;
  const ntrend = regression === "ct" ? 2 : 1;
  const dx = x.slice(1).map((v, i) => v - x[i]!);

  let usedLag: number;
  let lagSelection: AdfResult["lagSelection"];
  if (opts.lags !== undefined) {
    usedLag = opts.lags;
    lagSelection = "fixed";
  } else {
    let maxLag = opts.maxLag ?? Math.ceil(12 * (n / 100) ** 0.25);
    maxLag = Math.max(0, Math.min(maxLag, Math.floor(n / 2) - ntrend - 1));
    // Compare all candidate lags on a common sample (the one available at maxLag).
    let best = { aic: Infinity, lag: 0 };
    for (let p = 0; p <= maxLag; p++) {
      const { y, X } = design(x, dx, maxLag, maxLag, regression);
      const cols = ntrend + 1 + p;
      const fit = ols(
        y,
        X.map((r) => r.slice(0, cols)),
      );
      if (fit.aic < best.aic) best = { aic: fit.aic, lag: p };
    }
    usedLag = best.lag;
    lagSelection = "aic";
  }

  if (dx.length - usedLag <= ntrend + 1 + usedLag) {
    throw new Error(`Series too short (${n} observations) for an ADF regression with ${usedLag} lags.`);
  }
  const { y, X, levelIndex } = design(x, dx, usedLag, usedLag, regression);
  const fit = ols(y, X);
  const statistic = fit.coef[levelIndex]! / fit.stdErr[levelIndex]!;
  return {
    statistic,
    pValue: mackinnonPValue(statistic, regression),
    usedLag,
    nobs: fit.nobs,
    criticalValues: mackinnonCrit(regression, fit.nobs),
    regression,
    lagSelection,
  };
}
