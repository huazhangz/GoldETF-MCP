/**
 * Derivative-free minimization (Nelder-Mead with adaptive coefficients, Gao & Han 2012)
 * plus finite-difference gradient / Hessian helpers for standard errors.
 */

export interface MinimizeResult {
  x: number[];
  fx: number;
  evaluations: number;
  converged: boolean;
}

export interface MinimizeOptions {
  /** Initial simplex edge length (same for every coordinate). Default 0.25. */
  step?: number;
  maxEvaluations?: number;
  /** Convergence when simplex function spread < ftol * (|f| + 1e-8). Default 1e-12. */
  ftol?: number;
  /** Number of restarts from the best point with a fresh simplex. Default 2. */
  restarts?: number;
}

function nelderMead(f: (x: number[]) => number, x0: number[], step: number, maxEval: number, ftol: number) {
  const n = x0.length;
  const alpha = 1;
  const gamma = 1 + 2 / n;
  const rho = 0.75 - 1 / (2 * n);
  const sigma = 1 - 1 / n;
  let evals = 0;
  const F = (x: number[]) => {
    evals++;
    const v = f(x);
    return Number.isFinite(v) ? v : Infinity;
  };
  let simplex = [x0, ...x0.map((_, i) => x0.map((v, j) => (i === j ? v + step : v)))];
  let values = simplex.map(F);
  let converged = false;
  while (evals < maxEval) {
    const order = values.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
    simplex = order.map(([, i]) => simplex[i]!);
    values = order.map(([v]) => v);
    const best = values[0]!;
    const worst = values[n]!;
    if (Number.isFinite(worst) && Math.abs(worst - best) <= ftol * (Math.abs(best) + 1e-8)) {
      converged = true;
      break;
    }
    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j]! += simplex[i]![j]! / n;
    const at = (t: number) => centroid.map((c, j) => c + t * (simplex[n]![j]! - c));
    const xr = at(-alpha);
    const fr = F(xr);
    if (fr < values[0]!) {
      const xe = at(-alpha * gamma);
      const fe = F(xe);
      if (fe < fr) {
        simplex[n] = xe;
        values[n] = fe;
      } else {
        simplex[n] = xr;
        values[n] = fr;
      }
    } else if (fr < values[n - 1]!) {
      simplex[n] = xr;
      values[n] = fr;
    } else {
      const outside = fr < values[n]!;
      const xc = at(outside ? -alpha * rho : rho);
      const fc = F(xc);
      if (fc < (outside ? fr : values[n]!)) {
        simplex[n] = xc;
        values[n] = fc;
      } else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[0]!.map((b, j) => b + sigma * (simplex[i]![j]! - b));
          values[i] = F(simplex[i]!);
        }
      }
    }
  }
  const iBest = values.indexOf(Math.min(...values));
  return { x: simplex[iBest]!, fx: values[iBest]!, evals, converged };
}

export function minimize(f: (x: number[]) => number, x0: number[], opts: MinimizeOptions = {}): MinimizeResult {
  const n = x0.length;
  if (n === 0) return { x: [], fx: f([]), evaluations: 1, converged: true };
  const maxEval = opts.maxEvaluations ?? 4000 * n;
  const ftol = opts.ftol ?? 1e-12;
  let step = opts.step ?? 0.25;
  let res = nelderMead(f, x0, step, maxEval, ftol);
  let total = res.evals;
  let converged = res.converged;
  for (let r = 0; r < (opts.restarts ?? 2); r++) {
    step /= 4;
    const again = nelderMead(f, res.x, step, maxEval, ftol);
    total += again.evals;
    const improved = again.fx < res.fx - 1e-12 * (Math.abs(res.fx) + 1);
    if (again.fx <= res.fx) res = again;
    converged = again.converged;
    if (!improved) break;
  }
  return { x: res.x, fx: res.fx, evaluations: total, converged };
}

function stepFor(v: number, rel: number): number {
  return rel * Math.max(Math.abs(v), 1e-2);
}

/** Central-difference Hessian of a scalar function. */
export function numericHessian(f: (x: number[]) => number, x: number[], rel = 1e-4): number[][] {
  const n = x.length;
  const h = x.map((v) => stepFor(v, rel));
  const shift = (d: [number, number][]) => {
    const y = [...x];
    for (const [i, s] of d) y[i]! += s;
    return f(y);
  };
  const f0 = f(x);
  const H = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    H[i]![i] = (shift([[i, h[i]!]]) - 2 * f0 + shift([[i, -h[i]!]])) / (h[i]! * h[i]!);
    for (let j = 0; j < i; j++) {
      const v =
        (shift([[i, h[i]!], [j, h[j]!]]) -
          shift([[i, h[i]!], [j, -h[j]!]]) -
          shift([[i, -h[i]!], [j, h[j]!]]) +
          shift([[i, -h[i]!], [j, -h[j]!]])) /
        (4 * h[i]! * h[j]!);
      H[i]![j] = v;
      H[j]![i] = v;
    }
  }
  return H;
}

/** Central-difference Jacobian of a vector function (rows = outputs, cols = parameters). */
export function numericJacobian(f: (x: number[]) => number[], x: number[], rel = 1e-5): number[][] {
  const cols = x.map((v, i) => {
    const h = stepFor(v, rel);
    const up = [...x];
    const dn = [...x];
    up[i]! += h;
    dn[i]! -= h;
    const fu = f(up);
    const fd = f(dn);
    return fu.map((u, k) => (u - fd[k]!) / (2 * h));
  });
  const m = cols[0]?.length ?? 0;
  return Array.from({ length: m }, (_, r) => cols.map((c) => c[r]!));
}
