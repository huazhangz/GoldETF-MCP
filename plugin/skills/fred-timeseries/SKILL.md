---
name: fred-timeseries
description: Analyze and forecast US economic and market time series from FRED (inflation, unemployment, rates, yields, volatility indices such as VIX/GVZ, housing, production) with the fred-timeseries MCP tools. Use when the user asks whether a series is stationary, seasonal, volatile or forecastable, how two series are related (correlation, lead/lag, Granger), wants an ARIMA/GARCH model or forecast, or wants to know whether a model beats a naive forecast.
---

# FRED time-series analysis

The `fred-timeseries` MCP server provides FRED data (`fred_*` tools) and statistical tools (`ts_*`). The tools compute the statistics; your job is to pick the right question, read the verdicts they return, and report honestly. The fastest way to give a wrong answer here is to fit the model the user named without checking whether the data support it.

## Workflow

1. **Identify the series.** If the ID is uncertain, call `fred_search_series`. If a tool says `not_found`, search and retry with the correct ID; don't stop at the error. Check `fred_get_series_info` when units, frequency or seasonal adjustment matter.
2. **Fix the sample.** Pass explicit `start`/`end` dates whenever the user gives them. Aggregate with `frequency` (`m`, `q`, ...) rather than fetching daily data and averaging it yourself. Aggregation drops partially observed periods by default. For "so far this month", pass `keep_incomplete_periods: true` and say the figure is month-to-date.
3. **Choose the scale.** Use `transform: "log"` for positive, right-skewed levels (volatility indices, prices, counts). Use `log_diff` for growth rates such as month-on-month inflation from a price index. Year-over-year changes of monthly data are `log_seasonal_diff`.
4. **Diagnose before modeling.**
   - `ts_decompose_stl`: is seasonality worth modeling (F_S near or above 0.6), or negligible?
   - `ts_test_stationarity` with several `transforms` in one call. A `conflicting` verdict on a level usually means a persistent, mean-reverting series. Compare d=0 and d=1 instead of differencing by default.
   - `ts_autocorrelation`: ACF/PACF for model orders, ARCH-LM for volatility clustering, Jarque-Bera for fat tails.
5. **Relate two series** with `ts_compare_series`, on changes (`log_diff` for prices/indices, `diff` for rates), not trending levels. Report the autocorrelation-adjusted p-value, the lead/lag, and Granger results as "helps predict", never as causation.
6. **Model.** `ts_fit_arima` ranks candidates by AICc within each d and flags problems. Use `ts_fit_garch` only when ARCH effects are present. Use `variance: "constant", distribution: "t"` for fat tails without clustering.
7. **Validate before claiming skill.** Any statement that a model "forecasts better", "beats a random walk" or "is predictive" requires `ts_backtest` with a random walk as the first model.

## Rules that the tool outputs enforce. Follow them in your answer.

- **GARCH:** if ARCH-LM does not reject, or `ts_fit_garch` warns that alpha is on its lower bound, say GARCH is not warranted at this frequency. Do not report its persistence or half-life as findings. Clustering often exists in daily data and disappears in monthly averages.
- **Forecast claims:** report the backtest's verdict, including when it is negative. A lower RMSE with Diebold-Mariano p ≥ 0.05 is not evidence of skill. When a model *does* beat the benchmark significantly, say so with the horizon, the relative RMSE and the p-value.
- **Model choice:** do not pick an ARIMA because it is the only one whose Ljung-Box test passes, and do not compare AICc across different d. Heed warnings about insignificant terms, cancelling AR/MA roots, and over-differencing (an MA root near 1 after differencing).
- **Extreme recent shocks:** if `ts_fit_arima` warns that the last observations are extreme, present the forecast as unreliable. Compare it with a random walk or a Student-t model.
- **Intervals:** if residuals are non-normal, say that Gaussian intervals understate tail risk, or use Student-t errors.
- **Spurious relationships:** if `ts_compare_series` warns that a series is non-stationary, the level correlation is not evidence of a relationship. Say so and use the result on changes. Use `p_value_adjusted`, not `p_value_naive`.
- **Revisions:** macro series (GDP, payrolls, CPI) are revised. For "what was known at the time" questions, or honest historical backtests, pass `as_of` (an ALFRED vintage date).
- **Scope:** FRED has daily closes at most, with no intraday data, and these tools have no multivariate model beyond two series. Say what is not available rather than estimating it. Never copy long arrays of values between tool calls; pass `series_id` and let the tool fetch the data.
- **Numbers:** every number you report should come from a tool result, rounded sensibly. Back-transformed log forecasts are medians; say "median" when it matters.

## Reporting

Lead with the answer (yes/no, which model, what forecast), then the two or three numbers that support it, then the caveat that matters most. Name the series ID, the sample and the transform you used.

Not investment advice. This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis. Some series are copyrighted by third parties (e.g. CBOE volatility indices; see `fred_get_series_info`). Their owners' terms apply beyond personal use.
