/**
 * Minimal distribution functions needed by the diagnostics.
 * Implemented locally to keep the server dependency-free beyond the MCP SDK.
 */

/** Complementary error function via erfc(z) = Q(1/2, z^2); accurate to ~1e-15. */
function erfc(x: number): number {
  const r = x === 0 ? 1 : gammaQ(0.5, x * x);
  return x >= 0 ? r : 2 - r;
}

/** Standard normal CDF. */
export function normalCdf(x: number): number {
  return 0.5 * erfc(-x / Math.SQRT2);
}

/** Natural log of the gamma function (Lanczos approximation). */
export function logGamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = c[0]!;
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i]! / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x). */
function gammaQ(a: number, x: number): number {
  if (x < 0 || a <= 0) throw new RangeError("gammaQ: invalid arguments");
  if (x === 0) return 1;
  const gln = logGamma(a);
  if (x < a + 1) {
    // Series for P, then complement.
    let sum = 1 / a;
    let del = sum;
    let ap = a;
    for (let n = 0; n < 1000; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
    }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  // Continued fraction for Q (modified Lentz).
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h;
}

/** Survival function P(X > x) for a chi-square distribution with `df` degrees of freedom. */
export function chiSquareSf(x: number, df: number): number {
  if (df <= 0) throw new RangeError("chiSquareSf: df must be positive");
  if (x <= 0) return 1;
  return gammaQ(df / 2, x / 2);
}

/** Standard normal density. */
export function normalPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/** Standard normal quantile (Acklam's rational approximation + one Newton step). */
export function normalQuantile(p: number): number {
  if (p <= 0 || p >= 1) throw new RangeError("normalQuantile: p must be in (0, 1)");
  const a = [-39.69683028665376, 220.9460984245205, -275.928510446969, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  let x: number;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  } else if (p <= 1 - lo) {
    const q = p - 0.5;
    const r = q * q;
    x = ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) / (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  const e = normalCdf(x) - p;
  return x - e / normalPdf(x);
}

/** Continued fraction for the regularized incomplete beta (Numerical Recipes betacf). */
function betacf(a: number, b: number, x: number): number {
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}

/** Regularized incomplete beta I_x(a, b). */
export function betaInc(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  const bt = Math.exp(lbt);
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** CDF of Student's t with `df` degrees of freedom. */
export function studentTCdf(t: number, df: number): number {
  const x = df / (df + t * t);
  const tail = 0.5 * betaInc(x, df / 2, 0.5);
  return t >= 0 ? 1 - tail : tail;
}

/** Quantile of Student's t (bisection on the CDF; ~1e-12 accuracy). */
export function studentTQuantile(p: number, df: number): number {
  if (p <= 0 || p >= 1) throw new RangeError("studentTQuantile: p must be in (0, 1)");
  if (p === 0.5) return 0;
  let lo = -1;
  let hi = 1;
  while (studentTCdf(lo, df) > p) lo *= 2;
  while (studentTCdf(hi, df) < p) hi *= 2;
  for (let i = 0; i < 200 && hi - lo > 1e-12 * Math.max(1, Math.abs(lo)); i++) {
    const mid = 0.5 * (lo + hi);
    if (studentTCdf(mid, df) < p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Log density of the *standardized* Student-t (unit variance, nu > 2), evaluated at
 * e with variance sigma2 — the parameterization used by arch's StudentsT.
 */
export function stdTLogPdf(e: number, sigma2: number, nu: number): number {
  return (
    logGamma((nu + 1) / 2) -
    logGamma(nu / 2) -
    0.5 * Math.log(Math.PI * (nu - 2)) -
    0.5 * Math.log(sigma2) -
    ((nu + 1) / 2) * Math.log(1 + (e * e) / (sigma2 * (nu - 2)))
  );
}

/** Gaussian log density of e with variance sigma2. */
export function normalLogPdf(e: number, sigma2: number): number {
  return -0.5 * (Math.log(2 * Math.PI) + Math.log(sigma2) + (e * e) / sigma2);
}
