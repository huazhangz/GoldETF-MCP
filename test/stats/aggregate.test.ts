import { describe, expect, it } from "vitest";
import { aggregate } from "../../src/stats/aggregate.js";

const businessDays = (from: string, to: string) => {
  const out: { date: string; value: number }[] = [];
  for (let d = new Date(`${from}T00:00:00Z`); d <= new Date(`${to}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + 1)) {
    if (d.getUTCDay() % 6 !== 0) out.push({ date: d.toISOString().slice(0, 10), value: d.getUTCDate() });
  }
  return out;
};

describe("aggregate", () => {
  it("averages daily data by month and drops a partial last month", () => {
    const r = aggregate(businessDays("2008-06-03", "2008-09-17"), "m", "avg");
    expect(r.points.map((p) => p.date)).toEqual(["2008-06-01", "2008-07-01", "2008-08-01"]);
    expect(r.droppedIncomplete).toEqual(["2008-09-01"]);
  });

  it("treats a month ending on a weekend as complete", () => {
    // 2026-02-28 is a Saturday; last business day Friday 27th.
    const r = aggregate(businessDays("2026-01-01", "2026-02-27"), "m", "avg");
    expect(r.droppedIncomplete).toEqual([]);
    expect(r.points).toHaveLength(2);
  });

  it("detects a partial first quarter for monthly data", () => {
    const monthly = ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01", "2020-07-01"].map((date, i) => ({ date, value: i }));
    const r = aggregate(monthly, "q", "avg");
    expect(r.droppedIncomplete).toEqual(["2020-01-01", "2020-07-01"]);
    expect(r.points).toEqual([{ date: "2020-04-01", value: 3 }]);
  });

  it("supports end-of-period and sums, and can keep partial periods", () => {
    const pts = businessDays("2024-01-01", "2024-01-31");
    expect(aggregate(pts, "m", "eop").points[0]!.value).toBe(31);
    expect(aggregate(pts, "m", "sum").points[0]!.value).toBe(pts.reduce((s, p) => s + p.value, 0));
    expect(aggregate(businessDays("2024-01-01", "2024-02-10"), "m", "avg", { dropIncomplete: false }).points).toHaveLength(2);
  });

  it("dates weekly periods at the Friday", () => {
    const r = aggregate(businessDays("2024-01-01", "2024-01-12"), "w", "avg");
    expect(r.points.map((p) => p.date)).toEqual(["2024-01-05", "2024-01-12"]);
  });
});
