/**
 * Calendar aggregation of an observed series to a lower frequency, with detection of
 * incomplete first/last periods (e.g. the current month when data end on the 20th).
 *
 * Output dates follow FRED's conventions: monthly/quarterly/semiannual/annual periods are
 * dated at the period start; weekly periods end on Friday and are dated at the Friday.
 */
import type { Point } from "./transforms.js";

export type Frequency = "w" | "m" | "q" | "sa" | "a";
export type Aggregation = "avg" | "sum" | "eop";

const DAY = 86_400_000;
const toDate = (s: string) => new Date(`${s}T00:00:00Z`);
const iso = (d: Date) => d.toISOString().slice(0, 10);

function periodOf(date: Date, f: Frequency): { key: string; start: Date; end: Date } {
  const y = date.getUTCFullYear();
  const m = date.getUTCMonth();
  if (f === "w") {
    const toFri = (5 - date.getUTCDay() + 7) % 7;
    const end = new Date(date.getTime() + toFri * DAY);
    return { key: iso(end), start: new Date(end.getTime() - 6 * DAY), end };
  }
  const span = f === "m" ? 1 : f === "q" ? 3 : f === "sa" ? 6 : 12;
  const sm = Math.floor(m / span) * span;
  const start = new Date(Date.UTC(y, sm, 1));
  const end = new Date(Date.UTC(y, sm + span, 0));
  return { key: iso(start), start, end };
}

export interface AggregateResult {
  points: Point[];
  /** Periods dropped because they were only partially observed. */
  droppedIncomplete: string[];
}

export function aggregate(
  points: Point[],
  f: Frequency,
  how: Aggregation,
  opts: { dropIncomplete?: boolean } = {},
): AggregateResult {
  if (points.length === 0) return { points: [], droppedIncomplete: [] };
  const groups = new Map<string, { start: Date; end: Date; values: number[] }>();
  for (const p of points) {
    const per = periodOf(toDate(p.date), f);
    let g = groups.get(per.key);
    if (!g) {
      g = { start: per.start, end: per.end, values: [] };
      groups.set(per.key, g);
    }
    g.values.push(p.value);
  }
  // Native spacing (median gap in days) decides how much slack a "complete" period gets.
  const gaps: number[] = [];
  for (let i = 1; i < Math.min(points.length, 400); i++) {
    gaps.push((toDate(points[i]!.date).getTime() - toDate(points[i - 1]!.date).getTime()) / DAY);
  }
  gaps.sort((a, b) => a - b);
  const stepDays = gaps.length ? gaps[Math.floor(gaps.length / 2)]! : 1;
  const step = stepDays * DAY;
  const tol = 3 * DAY; // weekends and holidays
  const first = toDate(points[0]!.date).getTime();
  const last = toDate(points.at(-1)!.date).getTime();

  const out: Point[] = [];
  const dropped: string[] = [];
  const keys = [...groups.keys()];
  keys.forEach((key, i) => {
    const g = groups.get(key)!;
    // First period: complete if observed from (about) its start. Last period: complete if the last
    // observation, plus one native step, reaches (about) the period end.
    const incomplete =
      (i === 0 && first > g.start.getTime() + Math.max(tol, step)) ||
      (i === keys.length - 1 && last + step < g.end.getTime() + DAY - tol);
    if (incomplete && (opts.dropIncomplete ?? true)) {
      dropped.push(key);
      return;
    }
    const v = g.values;
    const value = how === "sum" ? v.reduce((s, x) => s + x, 0) : how === "eop" ? v.at(-1)! : v.reduce((s, x) => s + x, 0) / v.length;
    out.push({ date: key, value });
  });
  return { points: out, droppedIncomplete: dropped };
}
