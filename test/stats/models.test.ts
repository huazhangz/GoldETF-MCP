/**
 * Parity with statsmodels 0.15 / arch 8 / scipy (fixtures from scripts/gen_reference.py).
 *
 * Two kinds of check, kept separate on purpose:
 *  - likelihood / forecast at the *reference* parameters must match to ~1e-8: this verifies the
 *    model equations (Kalman filter, GARCH recursion, densities, backcast) independent of optimizers;
 *  - our own optimum must be at least as good as the reference optimum (loglik >= ref - 1e-4).
 */
import { describe, expect, it } from "vitest";
import arimaRef from "../fixtures/arima_reference.json" with { type: "json" };
import archRef from "../fixtures/arch_reference.json" with { type: "json" };
import misc from "../fixtures/misc_reference.json" with { type: "json" };
import { arimaAtParams, fitArima, forecastArima, arToPacf, pacfToAr } from "../../src/stats/arima.js";
import { fitVolModel, forecastVol, volLoglikAt, type VolSpec } from "../../src/stats/volatility.js";
import { stl } from "../../src/stats/stl.js";
import { archLm, jarqueBera } from "../../src/stats/diagnostics.js";
import { normalQuantile, studentTCdf, studentTQuantile } from "../../src/stats/distributions.js";
import { dieboldMariano } from "../../src/stats/evaluation.js";
import { backtest } from "../../src/stats/backtest.js";

const maxAbsDiff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i]!)));

describe("ARIMA vs statsmodels", () => {
  describe.each(arimaRef.map((c) => [c.order.join(","), c] as const))("ARIMA(%s)", (_label, c) => {
    const [p, d, q] = c.order as [number, number, number];
    const spec = { p, d, q, constant: true };
    const pr = c.params;
    const params = { mu: pr[0]!, ar: pr.slice(1, 1 + p), ma: pr.slice(1 + p, 1 + p + q), sigma2: pr.at(-1)! };

    it("log-likelihood, information criteria and residuals at statsmodels' parameters", () => {
      const f = arimaAtParams(c.y, spec, params);
      expect(f.loglik).toBeCloseTo(c.llf, 6);
      expect(f.aic).toBeCloseTo(c.aic, 6);
      expect(f.aicc).toBeCloseTo(c.aicc, 6);
      expect(f.bic).toBeCloseTo(c.bic, 6);
      expect(maxAbsDiff(f.residuals.slice(5), c.resid.slice(5))).toBeLessThan(1e-6);
    });

    it("forecast mean and variance at statsmodels' parameters", () => {
      const fc = forecastArima(arimaAtParams(c.y, spec, params), 8);
      expect(maxAbsDiff(fc.mean, c.fmean)).toBeLessThan(1e-9);
      expect(maxAbsDiff(fc.sd.map((s) => s * s), c.fvar)).toBeLessThan(1e-9);
    });

    it("own MLE is at least as good, with Hessian standard errors close to statsmodels'", () => {
      const f = fitArima(c.y, spec);
      expect(f.loglik).toBeGreaterThan(c.llf - 1e-4);
      if (Math.abs(f.loglik - c.llf) < 1e-3) {
        f.coefficients.forEach((co, i) => {
          expect(co.estimate).toBeCloseTo(pr[i]!, 2);
          expect(co.stdErr!).toBeGreaterThan(0);
          expect(Math.abs(co.stdErr! / c.bse_hessian[i]! - 1)).toBeLessThan(0.05);
        });
      }
    });
  });

  it("PACF <-> AR transform round-trips", () => {
    const phi = pacfToAr([0.5, -0.3, 0.2]);
    expect(arToPacf(phi)!.map((v) => Number(v.toFixed(12)))).toEqual([0.5, -0.3, 0.2]);
  });

  it("warns about near-cancelling roots and insignificant terms", () => {
    // White noise plus an over-specified ARMA(1,1): AR and MA roots should nearly cancel.
    let s = 1;
    const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
    const y = Array.from({ length: 300 }, rand);
    const f = fitArima(y, { p: 1, d: 0, q: 1, constant: true });
    expect(f.warnings.join(" ")).toMatch(/not significant|cancel/);
  });
});

describe("volatility models vs arch", () => {
  const y = archRef.y;
  describe.each(archRef.cases.map((c) => [`${c.kw.mean}-${c.kw.vol}-${c.kw.dist}`, c] as const))("%s", (_label, c) => {
    const spec: VolSpec = {
      mean: c.kw.mean === "AR" ? "ar" : c.kw.mean === "Constant" ? "constant" : "zero",
      lags: (c.kw as { lags?: number }).lags ?? 0,
      variance: c.kw.vol === "GARCH" ? "garch" : "constant",
      dist: c.kw.dist === "t" ? "t" : "normal",
    };
    const P = Object.fromEntries(c.names.map((n, i) => [n, c.params[i]!]));
    const params = {
      mu: P["Const"] ?? P["mu"] ?? 0,
      phi: P["y[1]"] !== undefined ? [P["y[1]"]!] : [],
      omega: (P["omega"] ?? P["sigma2"])!,
      alpha: P["alpha[1]"] ?? 0,
      beta: P["beta[1]"] ?? 0,
      nu: P["nu"] ?? Infinity,
    };

    it("log-likelihood at arch's parameters", () => {
      expect(volLoglikAt(y, spec, params)).toBeCloseTo(c.llf, 7);
    });

    it("own MLE, robust standard errors and forecasts", () => {
      const fit = fitVolModel(y, spec);
      expect(fit.loglik).toBeGreaterThan(c.llf - 1e-4);
      fit.coefficients.forEach((co, i) => {
        expect(Math.abs(co.estimate - c.params[i]!)).toBeLessThan(2e-3 * Math.max(1, Math.abs(c.params[i]!)) * 10);
        expect(Math.abs(co.stdErr! / c.se[i]! - 1)).toBeLessThan(0.05);
      });
      const fc = forecastVol({ ...fit, params }, 5);
      // arch's `variance` is the h-step forecast-error variance (AR propagation included).
      expect(maxAbsDiff(fc.mean, c.fmean)).toBeLessThan(1e-3);
      expect(Math.abs(fc.residualVariance[4]! / c.rvar[4]! - 1)).toBeLessThan(5e-3);
      expect(Math.abs(fc.sd[4]! ** 2 / c.fvar[4]! - 1)).toBeLessThan(5e-3);
    });
  });
});

describe("STL vs statsmodels", () => {
  const x = misc.stl.x;
  it.each([
    ["periodic_robust", { period: 12, robust: true }],
    ["periodic_plain", { period: 12, robust: false }],
    ["s13_robust", { period: 12, seasonal: 13, robust: true }],
  ] as const)("%s", (key, opts) => {
    const r = stl(x, opts);
    const ref = misc.stl[key];
    expect(maxAbsDiff(r.seasonal, ref.seasonal)).toBeLessThan(1e-10);
    expect(maxAbsDiff(r.trend, ref.trend)).toBeLessThan(1e-10);
    expect(maxAbsDiff(r.weights, ref.weights)).toBeLessThan(1e-10);
  });
});

describe("diagnostics and distributions vs statsmodels / scipy", () => {
  const dx = misc.stl.x.slice(1).map((v, i) => v - misc.stl.x[i]!);
  const m = dx.reduce((s, v) => s + v, 0) / dx.length;

  it("ARCH-LM", () => {
    const r = archLm(
      dx.map((v) => v - m),
      6,
    );
    expect(r.statistic).toBeCloseTo(misc.archlm.lm, 8);
    expect(r.pValue).toBeCloseTo(misc.archlm.p, 8);
  });

  it("Jarque-Bera", () => {
    const r = jarqueBera(dx);
    expect(r.statistic).toBeCloseTo(misc.jb[0]!, 8);
    expect(r.pValue).toBeCloseTo(misc.jb[1]!, 8);
    expect(r.skewness).toBeCloseTo(misc.jb[2]!, 8);
    expect(r.kurtosis).toBeCloseTo(misc.jb[3]!, 8);
  });

  it("normal and Student-t quantiles / CDF", () => {
    for (const [p, q] of misc.dist.normal_q) expect(normalQuantile(p!)).toBeCloseTo(q!, 8);
    for (const [x, df, c] of misc.dist.t_cdf) expect(studentTCdf(x!, df!)).toBeCloseTo(c!, 10);
    for (const [p, df, q] of misc.dist.t_q) expect(studentTQuantile(p!, df!)).toBeCloseTo(q!, 8);
  });
});

describe("forecast evaluation", () => {
  it("Diebold-Mariano: identical errors -> no test; clearly better model -> significant", () => {
    const e = Array.from({ length: 100 }, (_, i) => Math.sin(i) * (1 + (i % 7)));
    expect(dieboldMariano(e, e, 1)).toBeNull();
    const better = e.map((v) => v * 0.5);
    const r = dieboldMariano(better, e, 1)!;
    expect(r.meanDiff).toBeLessThan(0);
    expect(r.pValue).toBeLessThan(0.01);
  });

  it("backtest: on a strongly mean-reverting AR(1), AR(1) beats a random walk", () => {
    let s = 3;
    const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
    const y = [0];
    for (let t = 1; t < 260; t++) y.push(0.2 * y[t - 1]! + rand());
    const r = backtest(y, [{ kind: "random_walk" }, { kind: "arima", order: [1, 0, 0] }], { initial: 60, horizons: [1, 3] });
    const ar = r.models[1]!.byHorizon[1]!;
    expect(ar.relRmse).toBeLessThan(0.85); // theory: sqrt((1 + phi) / 2) = 0.775
    expect(ar.dm!.pValue).toBeLessThan(0.05);
    expect(r.models[0]!.byHorizon[1]!.n).toBe(200);
    expect(r.models[0]!.byHorizon[3]!.n).toBe(198);
  });
});
