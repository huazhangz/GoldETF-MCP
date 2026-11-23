import type { Deterministic } from "./adf.js";

export interface KpssResult {
  statistic: number;
  /** Interpolated from the KPSS (1992) table; only resolvable inside [0.01, 0.10]. */
  pValue: number;
  /** "lower" means the true p-value is below 0.01; "upper" means above 0.10. */
  pValueBound: "exact" | "lower" | "upper";
  lags: number;
  criticalValues: { "10%": number; "5%": number; "2.5%": number; "1%": number };
  regression: Deterministic;
}

const CRIT = {
  c: [0.347, 0.463, 0.574, 0.739],
  ct: [0.119, 0.146, 0.176, 0.216],
} as const;
const PVALS = [0.1, 0.05, 0.025, 0.01];

export interface KpssOptions {
  regression?: Deterministic;
  /** Newey-West bandwidth. Default trunc(4 * (n/100)^(1/4)), the R `tseries::kpss.test` short rule. */
  lags?: number;
}

/** KPSS test. H0: the series is (level- or trend-) stationary. */
export function kpssTest(x: number[], opts: KpssOptions = {}): KpssResult {
  const regression = opts.regression ?? "c";
  const n = x.length;
  if (n < 10) throw new Error(`KPSS needs at least 10 observations; got ${n}.`);
  const lags = opts.lags ?? Math.trunc(4 * (n / 100) ** 0.25);
  if (lags >= n) throw new RangeError(`lags (${lags}) must be smaller than the series length (${n}).`);

  // Residuals from regressing on a constant (level) or constant + trend.
  let resid: number[];
  if (regression === "c") {
    const m = x.reduce((s, v) => s + v, 0) / n;
    resid = x.map((v) => v - m);
  } else {
    const tMean = (n + 1) / 2;
    const yMean = x.reduce((s, v) => s + v, 0) / n;
    let sxy = 0;
    let sxx = 0;
    for (let i = 0; i < n; i++) {
      sxy += (i + 1 - tMean) * (x[i]! - yMean);
      sxx += (i + 1 - tMean) ** 2;
    }
    const b = sxy / sxx;
    const a = yMean - b * tMean;
    resid = x.map((v, i) => v - a - b * (i + 1));
  }

  let cum = 0;
  let eta = 0;
  for (const e of resid) {
    cum += e;
    eta += cum * cum;
  }
  eta /= n * n;

  let s2 = resid.reduce((s, e) => s + e * e, 0);
  for (let k = 1; k <= lags; k++) {
    let gk = 0;
    for (let t = k; t < n; t++) gk += resid[t]! * resid[t - k]!;
    s2 += 2 * (1 - k / (lags + 1)) * gk;
  }
  s2 /= n;
  const statistic = eta / s2;

  const crit = CRIT[regression];
  let pValue: number;
  let pValueBound: KpssResult["pValueBound"] = "exact";
  if (statistic <= crit[0]) {
    pValue = 0.1;
    pValueBound = "upper";
  } else if (statistic >= crit[3]) {
    pValue = 0.01;
    pValueBound = "lower";
  } else {
    let i = 0;
    while (statistic > crit[i + 1]!) i++;
    const w = (statistic - crit[i]!) / (crit[i + 1]! - crit[i]!);
    pValue = PVALS[i]! + w * (PVALS[i + 1]! - PVALS[i]!);
  }

  return {
    statistic,
    pValue,
    pValueBound,
    lags,
    criticalValues: { "10%": crit[0], "5%": crit[1], "2.5%": crit[2], "1%": crit[3] },
    regression,
  };
}
