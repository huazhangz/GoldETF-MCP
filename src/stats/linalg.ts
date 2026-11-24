/** Small dense linear algebra helpers (row-major number[][]). Sizes here are tiny (<= ~40). */

export type Matrix = number[][];

export function zeros(r: number, c: number): Matrix {
  return Array.from({ length: r }, () => new Array<number>(c).fill(0));
}

export function identity(n: number): Matrix {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

/** Solve A X = B by Gauss-Jordan elimination with partial pivoting. B may have several columns. */
export function solve(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const m = B[0]?.length ?? 0;
  const tol = 1e-12 * Math.max(...A.map((row, i) => Math.abs(row[i]!)), 1e-300);
  const a = A.map((row, i) => [...row, ...B[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r]![col]!) > Math.abs(a[pivot]![col]!)) pivot = r;
    if (Math.abs(a[pivot]![col]!) < tol) {
      throw new SingularMatrixError();
    }
    [a[col], a[pivot]] = [a[pivot]!, a[col]!];
    const p = a[col]![col]!;
    for (let j = col; j < n + m; j++) a[col]![j]! /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = a[r]![col]!;
      if (f === 0) continue;
      for (let j = col; j < n + m; j++) a[r]![j]! -= f * a[col]![j]!;
    }
  }
  return a.map((row) => row.slice(n));
}

export function invert(A: Matrix): Matrix {
  return solve(A, identity(A.length));
}

export class SingularMatrixError extends Error {
  constructor() {
    super(
      "Regression is singular: the series is constant or perfectly deterministic (e.g. an exact sine wave). " +
        "Check the data, or set a smaller fixed number of lags.",
    );
  }
}

export function matMul(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const k = B.length;
  const m = B[0]?.length ?? 0;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let l = 0; l < k; l++) {
      const v = A[i]![l]!;
      if (v === 0) continue;
      for (let j = 0; j < m; j++) out[i]![j]! += v * B[l]![j]!;
    }
  return out;
}

export function transpose(A: Matrix): Matrix {
  return A[0] ? A[0].map((_, j) => A.map((row) => row[j]!)) : [];
}

/** Roots of the polynomial c[0] + c[1] z + ... + c[n] z^n (Durand-Kerner). Returns [re, im] pairs. */
export function polyRoots(c: number[]): [number, number][] {
  let coef = [...c];
  while (coef.length > 1 && Math.abs(coef.at(-1)!) < 1e-14) coef.pop();
  const n = coef.length - 1;
  if (n < 1) return [];
  const lead = coef[n]!;
  const a = coef.map((v) => v / lead); // monic
  let roots: [number, number][] = Array.from({ length: n }, (_, k) => {
    const r = 1 + Math.max(...a.slice(0, n).map(Math.abs));
    const ang = (2 * Math.PI * k) / n + 0.4;
    return [r * Math.cos(ang) * 0.9, r * Math.sin(ang) * 0.9];
  });
  const evalP = (z: [number, number]): [number, number] => {
    let re = 1;
    let im = 0; // leading coefficient 1
    for (let i = n - 1; i >= 0; i--) {
      const nr = re * z[0] - im * z[1] + a[i]!;
      const ni = re * z[1] + im * z[0];
      re = nr;
      im = ni;
    }
    return [re, im];
  };
  for (let iter = 0; iter < 500; iter++) {
    let maxDelta = 0;
    roots = roots.map((z, i) => {
      let [nr, ni] = evalP(z);
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dr = z[0] - roots[j]![0];
        const di = z[1] - roots[j]![1];
        const den = dr * dr + di * di || 1e-300;
        const tr = (nr * dr + ni * di) / den;
        const ti = (ni * dr - nr * di) / den;
        nr = tr;
        ni = ti;
      }
      maxDelta = Math.max(maxDelta, Math.hypot(nr, ni));
      return [z[0] - nr, z[1] - ni];
    });
    if (maxDelta < 1e-14) break;
  }
  return roots;
}
