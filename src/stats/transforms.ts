export interface Point {
  date: string; // YYYY-MM-DD
  value: number;
}

export type Transform =
  | "none"
  | "log"
  | "diff"
  | "log_diff"
  | "seasonal_diff"
  | "log_seasonal_diff"
  | "log_diff_seasonal_diff";

/**
 * Apply a named transform to an ordered series. Differencing drops leading points,
 * and the date of each differenced point is the later of the two dates.
 */
export function applyTransform(points: Point[], transform: Transform, seasonalPeriod = 12): Point[] {
  const needsLog = transform.startsWith("log");
  if (needsLog) {
    const bad = points.find((p) => p.value <= 0);
    if (bad) {
      throw new Error(
        `Cannot take log: value ${bad.value} on ${bad.date} is not positive. Use transform "diff" or "none" instead.`,
      );
    }
  }
  let out = needsLog ? points.map((p) => ({ date: p.date, value: Math.log(p.value) })) : points;
  if (transform === "diff" || transform === "log_diff" || transform === "log_diff_seasonal_diff") {
    out = difference(out, 1);
  }
  if (transform === "seasonal_diff" || transform === "log_seasonal_diff" || transform === "log_diff_seasonal_diff") {
    out = difference(out, seasonalPeriod);
  }
  return out;
}

export function difference(points: Point[], lag: number): Point[] {
  return points.slice(lag).map((p, i) => ({ date: p.date, value: p.value - points[i]!.value }));
}

export function summarize(values: number[]) {
  const n = values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((s, v) => s + v, 0) / n;
  const sd = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, n - 1));
  const q = (p: number) => {
    const idx = (n - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (idx - lo);
  };
  return { n, mean, sd, min: sorted[0]!, p25: q(0.25), median: q(0.5), p75: q(0.75), max: sorted[n - 1]! };
}
