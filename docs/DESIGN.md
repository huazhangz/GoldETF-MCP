# Design notes

This server is a small exercise in **tool design for LLM agents**. These are the decisions and the reasons for them.

## 1. Naming

- `fred_*` for data access and `ts_*` for analysis. A client may load several servers at once, and a prefix
  makes it clear where a tool comes from and keeps names from colliding (a bare `search` is ambiguous).
- Names are verb + object in snake_case: `fred_get_observations`, `ts_test_stationarity`.
- Tool names and parameters reuse terms the model already knows from statistics and from the FRED docs
  (`series_id`, `frequency`, `regression: "c" | "ct"`), so there is no new vocabulary to learn.

## 2. Descriptions are prompts

Each description answers three questions: **what it does, when to use it, and what to call next.**

- `fred_search_series` says to use it first when the ID is unknown and gives an example mapping
  ("gold volatility" -> GVZCLS).
- `fred_get_observations` tells the model *not* to copy values into the analysis tools and to pass `series_id`
  instead.
- Server `instructions` describe the typical chain:
  search -> info -> stationarity -> autocorrelation.
- Parameter descriptions include valid ranges, defaults and examples. Enums replace free text wherever the
  input has a fixed set of values.

## 3. Keep data out of the context window

The model never has to move hundreds of numbers between calls.

- Analysis tools fetch the data themselves from `series_id`, and the FRED client caches responses for
  10 minutes, so a chain of calls hits FRED once.
- `fred_get_observations` always returns summary statistics, and returns at most `max_points` points
  (default 120). When it truncates, it says so and says how to get the rest.
- Numbers are rounded to 4-6 significant digits.
- `ts_test_stationarity` accepts `transforms: [...]` and compares them in **one** call instead of N.

## 4. Return conclusions as well as statistics

LLMs are bad at reading p-value tables consistently. The tools return the raw statistics *and* an
interpretation:

- `ts_test_stationarity` returns `stationary | non_stationary | conflicting | inconclusive` with a next step,
  for example "difference the series" or "retry with regression ct".
- `ts_autocorrelation` lists `significant_acf_lags` and `significant_pacf_lags` explicitly and warns that about
  5% of lags exceed the band by chance.
- When a KPSS p-value falls outside the lookup table, the result says whether the true value is above or
  below the reported one.

## 5. Errors the model can act on

Every failure comes back as `isError: true` with a message that says **what went wrong and what to do**.
None of them surface as a protocol error, which the client may hide from the model.

| Situation | Message gives the model |
| --- | --- |
| Unknown series ID | `not_found` + "Call fred_search_series to find the correct series ID." |
| No API key | where to get one and where to put it; notes that inline `values` still work |
| `series_id` and `values` both given | "Provide exactly one of ..." |
| `log` of a non-positive value | the offending date/value + a suggested alternative transform |
| Too few points after differencing | the count + "widen the date range" |
| Rate limit | `rate_limited` + "wait about a minute" |
| Bad date format | Zod schema error naming the field and `YYYY-MM-DD` |

The API key is kept out of the cache key and is never echoed back. A test asserts this.

## 6. Correctness

Statistics are implemented in plain TypeScript (no Python bridge), so installing the server only requires
`node`. To make that trustworthy, `test/stats/reference.test.ts` checks ADF (AIC lag selection and fixed lags,
`c`/`ct`), KPSS, ACF, PACF and Ljung-Box against statsmodels output on three synthetic series of length 215,
the same length as the monthly GVZ sample.

## Roadmap

- [ ] `ts_stl_decompose`: trend/seasonal strength (F_T, F_S), reproducing the project's STL section
- [ ] `ts_fit_arima`: fit candidate orders, rank by AICc, run residual Ljung-Box, return a forecast with intervals
- [ ] Rolling-origin backtest against naive / random-walk baselines (the original project had no out-of-sample check)
- [ ] `outputSchema` + `structuredContent` for typed results
- [ ] MCP resources for popular series (`fred://series/GVZCLS`) and a prompt template for the analysis workflow
- [ ] Tool-use evals: a fixed set of natural-language tasks scored on whether the model picks the right tools and arguments
- [ ] Publish to npm (`npx fred-timeseries-mcp`) and the MCP Registry
