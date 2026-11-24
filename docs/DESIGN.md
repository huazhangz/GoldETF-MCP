# Design notes

This server is an exercise in **tool design for LLM agents**. A statistics toolbox is a good test case because
it is easy for a model to *call* a method and hard for it to know *whether the method is warranted*. These are
the decisions and the reasons for them.

## 1. Naming

- `fred_*` for data access, `ts_*` for analysis. A client may load several servers at once. The prefix says where
  a tool comes from and avoids collisions with tools such as a bare `search`.
- Names are verb + object in snake_case: `ts_fit_arima`, `ts_test_stationarity`, `ts_backtest`.
- Tool names and parameters use terms the model already knows from statistics texts and the FRED docs
  (`series_id`, `order: [p, d, q]`, `regression: "c" | "ct"`, `distribution: "t"`).

## 2. Descriptions are prompts

Each description answers three questions: **what the tool does, when to use it, and what to call next.**

- The server `instructions` give the canonical chain: search -> info -> STL -> stationarity -> autocorrelation ->
  ARIMA / (only if ARCH effects) GARCH -> backtest.
- `fred_get_observations` tells the model *not* to copy values into the analysis tools.
- `ts_fit_arima` says that in-sample fit does not prove forecasting skill, and points to `ts_backtest`.
- Every parameter has a description with range, default and an example. A test enforces this for every tool.
- Enums replace free text where the input has a fixed set of values. Nested model specs use flat objects with
  a `kind` enum, not `anyOf` unions, which some clients render poorly.

## 3. Statistical guardrails: tools that say when *not* to do something

An agent with a GARCH tool will fit GARCH. The tools therefore return the evidence for and against the method,
not just the estimates. The case study shows each of these firing on real data.

| Guardrail | Where | What it prevents |
| --- | --- | --- |
| ARCH-LM pre-test and a constant-variance comparison (LR, ΔBIC) | `ts_fit_garch`, `ts_autocorrelation` | Fitting GARCH where there is no clustering (monthly GVZ: p = 0.994, ΔBIC +9) |
| "alpha on its lower bound: beta not identified" | `ts_fit_garch` | Reading a degenerate fit as "persistence 0.96" |
| AICc ranked **within** each d; d = 0 vs d = 1 sent to the backtest | `ts_fit_arima` | Comparing likelihoods of levels with likelihoods of differences |
| MA root near 1 when d ≥ 1: "likely over-differenced" | `ts_fit_arima` | Differencing a mean-reverting series by default |
| Near-cancelling AR/MA roots; insignificant terms | `ts_fit_arima` | Adding terms until Ljung-Box passes (the course project's ARIMA(2,1,2)) |
| `conflicting` ADF/KPSS verdict with a next step, not a forced choice | `ts_test_stationarity` | Treating a persistent mean-reverting series as a unit root |
| Jarque-Bera with an explicit "Gaussian intervals misstate tail risk" | `ts_autocorrelation`, `ts_fit_arima` | Reporting normal prediction intervals for fat-tailed data |
| Diebold-Mariano verdicts, plus shock concentration | `ts_backtest` | Claiming "beats the random walk" from a 2% RMSE difference |
| Partial last month dropped when aggregating | data layer | Averages of the current month that are not comparable to the others |
| "Extreme shock in the last observations" | `ts_fit_arima` | Forecasting May 2020 unemployment at 24% from a fit dominated by April 2020 (statsmodels gives the same fit) |

## 4. Keep data out of the context window

- Analysis tools fetch data from a `series_id` themselves. The FRED client caches responses for 10 minutes, so
  a chain of calls, or `ts_test_stationarity` with five transforms, hits FRED once.
- `fred_get_observations` always returns summary statistics plus at most `max_points` points. When it
  truncates, it says so.
- Numbers are rounded to 4-6 significant digits, and p-values below 1e-4 are reported as `"<1e-4"`.
- `ts_backtest` returns aggregate scores per horizon, never per-origin forecasts.

## 5. Errors the model can act on

Every failure comes back as `isError: true` with **what went wrong and what to do**. Nothing surfaces as a
protocol error that the client might hide.

| Situation | Message gives the model |
| --- | --- |
| Unknown series ID (API or CSV path) | `not_found` and "Call fred_search_series to find the correct series ID." |
| No API key | where to get a free key and where to put it; the opt-in `FRED_CSV_FALLBACK=1` for personal, data-only use |
| `series_id` and `values` both given | "Provide exactly one of ..." |
| `log` of a non-positive value | the offending date and value, and an alternative transform |
| Too few points after differencing or aggregation | the count, the minimum, and "widen the date range" |
| Backtest `arima` model without an `order` | `models[i]: kind "arima" needs an "order", e.g. [1, 0, 0]` |
| Too many fits requested | origins × models, and "raise step or drop models" |
| Rate limit | `rate_limited` and "wait about a minute" |
| Bad date format | the Zod schema error naming the field and `YYYY-MM-DD` |

The API key is kept out of the cache key and never echoed back; a test asserts this.

## 6. Correctness: two kinds of test

All numerics are plain TypeScript, so installing the server only requires `node`. To make that trustworthy,
`test/stats/` separates **model correctness** from **optimizer quality**:

1. **At the reference's own parameters**, our log-likelihood, residuals, information criteria and forecasts must
   equal statsmodels' / arch's. Measured differences are 1e-7 or smaller for ARIMA (Kalman filter) and around
   1e-13 for GARCH-t. This tests the equations (state space, backcast, standardized-t density) independently of
   any optimizer.
2. **Our own optimum** must be at least as good as the reference's: log-likelihood ≥ reference − 1e-4. This
   bound has to be one-sided. On GVZ, statsmodels stops 4.1 log-likelihood units short on ARIMA(1,1,1). Without
   rescaling, arch fails outright on daily GVZ, where it returns ν = 500 and α = 0.76. Internal unit-variance
   scaling avoids that.

STL, ADF (AIC lag selection), KPSS, ACF/PACF, Ljung-Box, ARCH-LM, Jarque-Bera and the normal and t quantiles
match to 1e-8 or better. All fixtures are seeded synthetic data from `scripts/gen_reference.py`. End-to-end
tests drive the real MCP server through an in-memory client, with FRED mocked.

Standard errors: ARIMA uses the observed information (numerical Hessian), as R's `arima` does. statsmodels
defaults to OPG. GARCH uses Bollerslev-Wooldridge robust (sandwich) errors, as arch does. When a parameter sits
on a boundary, it is held fixed and the rest are reported conditionally.

## 7. Performance

A backtest re-fits every model at every origin. Two changes keep `ts_backtest` interactive (about 6 s for 7
models × 143 origins):

- The Kalman filter runs on flat `Float64Array`s, freezes the covariance once it reaches steady state, and
  avoids allocation in the hot loop. This made ARIMA(2,1,2) fits 9× faster, from 428 ms to 48 ms.
- Backtests warm-start each fit from the previous origin's estimates and use a looser tolerance, which is
  adequate for forecasting.

## Roadmap

- [ ] Daily-frequency models aggregated to monthly forecasts (HAR, daily AR-GARCH-t by simulation). This is
      where the case study says the remaining signal is.
- [ ] Skewed-t innovations (Hansen 1994) for upward-spiking series
- [ ] Exogenous regressors (VIX, gold returns, real rates) in `ts_fit_arima` / `ts_backtest`
- [ ] `outputSchema` + `structuredContent` for typed results; MCP resources for popular series; a prompt template
      for the analysis workflow
- [x] Tool-use evals (`evals/`): 20 two-sided cases with deterministic graders, validated with oracle/null/naive
      scripted agents; variants for skill on/off, minimal descriptions, and guardrail text stripped
- [ ] Run the eval on Claude (pilot, then 3 reps per variant) and publish the variant comparison
- [x] Claude Code plugin + marketplace (`plugin/`, `.claude-plugin/marketplace.json`; validated with `claude plugin validate`)
- [ ] Publish to npm (`npx fred-timeseries-mcp`) and the MCP Registry
- [ ] Correlation / regression / Granger tools (the eval's out-of-scope case shows the gap)
