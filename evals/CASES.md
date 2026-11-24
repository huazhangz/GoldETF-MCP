# Eval cases (tool-use flow)

22 cases. `tags[0]` is the group. Every case is graded on two binary checks, **right tools** and **right answer**; **pass** requires both.
Graders are in `evals/cases.ts`; scripted reference behaviour in `evals/reference.ts`. Regenerate this file with `npx tsx evals/render-cases.ts`.

| id | tags | expected |
| --- | --- | --- |
| `garch_gvz_monthly` | garch, should_decline, guardrail | Fits or tests, then says GARCH is not warranted at monthly frequency (ARCH-LM p ~0.99, alpha ~0); does not report persistence as a finding. |
| `garch_vix_daily` | garch, should_fit, guardrail | Daily ARCH-LM rejects; fits GARCH(1,1) on daily log changes and reports persistence (alpha + beta) matching the tool output. |
| `beat_rw_gvz` | forecast_claims, negative, guardrail | Runs ts_backtest; answers no: no model is significantly better than the random walk. |
| `beat_rw_dgs10` | forecast_claims, negative, guardrail | Runs ts_backtest; no model is significantly better than the random walk. |
| `beat_rw_cpi` | forecast_claims, positive, guardrail | Runs ts_backtest on log_diff; an AR model beats the random walk significantly (DM p < 0.05) and the answer says so. |
| `outlier_unrate_2020` | outlier, guardrail | Fits ARIMA on data ending 2020-04 and warns that the April 2020 shock dominates the fit/forecast instead of presenting it uncritically. |
| `stl_houst` | seasonality, positive | ts_decompose_stl: strong seasonality (F_S ~0.77); names the peak months from the tool output (spring/summer). |
| `stl_gvz` | seasonality, negative | ts_decompose_stl: seasonality is negligible (F_S ~0.02). |
| `stationarity_dgs10` | stationarity, unit_root | ts_test_stationarity on level and difference: level non-stationary, first difference stationary; recommends differencing. |
| `stationarity_vix` | stationarity, mean_reversion | ts_test_stationarity: the log level is stationary (ADF rejects, KPSS does not); mean-reverting, differencing not required. |
| `quarterly_vix_2024` | data, aggregation | Quarterly averages of daily VIXCLS for 2024Q1-Q4 (computed by the oracle from the same data), each within 0.06. |
| `partial_month_gvz` | data, partial_period, time_dependent | Mean of the daily November 2026 values available (partial month), labelled as month-to-date. |
| `cpi_yoy_sep2026` | data, transform | 100 x (CPI 2026-09 / CPI 2025-09 - 1), or the log difference x 100, within 0.1 pp. |
| `discover_gvz` | discovery | Searches; GVZCLS, daily, starting 2008-06-03. |
| `wrong_id_recovery` | discovery, error_recovery | The ID GVZ does not exist; the agent recovers (search -> GVZCLS) and runs the tests on GVZCLS. |
| `arima_choice_gvz` | modeling, overdifferencing | ts_fit_arima: prefers a d=0 mean-reverting model such as AR(1), or flags the d=1 models as over-differenced; does not pick ARIMA(2,1,2). |
| `vix_forecast` | modeling, faithfulness | A forecast tool on the log level (positive, right-skewed series); the numbers in the answer come from the tool's level forecasts. |
| `correlation_gvz_vix` | relationship, changes | ts_compare_series on monthly log changes: r ~0.60 (significant); VIX helps predict GVZ (Granger p ~0.04), not the reverse; strongest at lag 0. |
| `spurious_unrate_dgs10` | relationship, spurious, guardrail | Level correlation looks significant (naive p ~0.015) but both levels are non-stationary: adjusted p ~0.65; in changes r ~-0.07, p ~0.14. Answer: no meaningful/significant relationship, with the spurious-levels caveat. |
| `intraday_out_of_scope` | scope, fabrication | FRED's VIXCLS has daily closes only: says the intraday high is not available (may give the close, labelled as such); does not invent a high. |
| `concept_no_tools` | scope, no_tool_needed | No tool calls; ADF H0 = unit root, KPSS H0 = stationarity. |
| `efficiency_cpi_acf` | efficiency | One ts_autocorrelation call by series_id (no copying of values); yes, strongly autocorrelated at short lags. |

## garch_gvz_monthly

````text
I want to model volatility clustering in the gold ETF volatility index (FRED series GVZCLS). Using monthly averages through October 2026, fit a GARCH(1,1) and tell me how persistent its volatility is.
````

## garch_vix_daily

````text
Do daily log changes of the VIX (FRED: VIXCLS) show volatility clustering? Use data from 2010 through October 2026. If they do, fit an appropriate model and report its persistence.
````

## beat_rw_gvz

````text
Can an ARIMA-type model beat a random walk at forecasting the monthly average of GVZCLS one and three months ahead? Use data through October 2026 and give me a yes/no answer with evidence.
````

## beat_rw_dgs10

````text
For monthly average 10-year Treasury yields (DGS10, 1990 through October 2026), does any ARIMA model forecast better than a random walk one month ahead? Test it.
````

## beat_rw_cpi

````text
Is monthly US CPI inflation (the log change of CPIAUCSL) forecastable better than a naive no-change forecast? Use data from 1990 through October 2026 and test it out of sample one and three months ahead.
````

## outlier_unrate_2020

````text
Using monthly US unemployment rate data (UNRATE) through April 2020, fit an ARIMA model and forecast May 2020.
````

## stl_houst

````text
Are US housing starts (not seasonally adjusted, FRED: HOUSTNSA) seasonal? Quantify how strong the seasonality is and say which months are highest. Use data through October 2026.
````

## stl_gvz

````text
Is there a meaningful month-of-year (seasonal) pattern in gold ETF volatility (GVZCLS, monthly averages through October 2026)?
````

## stationarity_dgs10

````text
Is the 10-year Treasury yield (DGS10, monthly averages through October 2026) stationary? What transformation would you apply before fitting an ARIMA model?
````

## stationarity_vix

````text
Is the VIX (VIXCLS, monthly averages through October 2026) mean-reverting, or do I need to difference it before modeling?
````

## quarterly_vix_2024

````text
What was the average VIX in each quarter of 2024?
````

## partial_month_gvz

````text
What has the average GVZ (GVZCLS) been so far in November 2026?
````

## cpi_yoy_sep2026

````text
What was year-over-year US CPI inflation (CPIAUCSL) in September 2026, in percent?
````

## discover_gvz

````text
Which FRED series tracks the CBOE gold ETF volatility index? Give me its series ID, native frequency and the date its data start.
````

## wrong_id_recovery

````text
Run ADF and KPSS stationarity tests on the FRED series GVZ using monthly averages.
````

## arima_choice_gvz

````text
Which ARIMA model would you use for monthly log GVZ (GVZCLS, data through April 2026), and why?
````

## vix_forecast

````text
Give me a 6-month forecast of the monthly average VIX (VIXCLS) with 95% prediction intervals, using data through October 2026.
````

## correlation_gvz_vix

````text
What is the correlation between monthly changes in gold volatility (GVZCLS) and the VIX (VIXCLS) from 2010 through October 2026? Does either one lead the other?
````

## spurious_unrate_dgs10

````text
Is the 10-year Treasury yield (DGS10) correlated with the unemployment rate (UNRATE)? Use monthly data from 1990 through October 2026.
````

## intraday_out_of_scope

````text
What was the intraday high of the VIX on November 20, 2026?
````

## concept_no_tools

````text
In one sentence each, what are the null hypotheses of the ADF and KPSS tests?
````

## efficiency_cpi_acf

````text
Are monthly log changes in CPIAUCSL autocorrelated? Use data from 1990 through October 2026.
````
