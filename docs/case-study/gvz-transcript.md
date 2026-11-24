# GVZ case study: tool-call transcript

Generated 2026-11-24 by `scripts/case_study.ts` (sample end 2026-04-30 unless noted).
Each step is one MCP tool call; full results are in `gvz-results.json`.

## 1. Data: monthly averages of daily GVZ

```json
fred_get_observations {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","max_points":0}
```

n = 215, 2008-06-01 to 2026-04-01; dropped missing = 165; partial periods dropped = [].

## 2. STL decomposition of log(GVZ)

```json
ts_decompose_stl {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log"}
```

F_T = 0.6862, F_S = 0.03044. Seasonality is negligible (F_S = 0.030; peak-to-trough 0.081 vs remainder RMS 0.172): seasonal terms are not worth modeling. Trend/cycle is strong (F_T = 0.686).

Largest shocks: 2008-06 (-0.9813), 2008-07 (-0.8151), 2020-03 (0.722), 2008-08 (-0.6946), 2020-04 (0.5606), 2011-09 (0.521)

## 3. Stationarity across transforms

```json
ts_test_stationarity {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transforms":["none","log","log_diff","log_seasonal_diff","log_diff_seasonal_diff"]}
```

| transform | ADF p | KPSS p | verdict |
|---|---|---|---|
| none | 0.001603 | 0.01 (true p-value < 0.01) | conflicting |
| log | 0.01212 | 0.01 (true p-value < 0.01) | conflicting |
| log_diff | 2.473e-17 | 0.1 (true p-value > 0.10) | stationary |
| log_seasonal_diff | 0.04635 | 0.03538346611227211 | conflicting |
| log_diff_seasonal_diff | 8.501e-11 | 0.1 (true p-value > 0.10) | stationary |

## 4. Reproduce R tseries::adf.test settings (trend + fixed 5 lags) on log(GVZ)

```json
ts_test_stationarity {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log","regression":"ct","adf_lags":5}
```

| transform | ADF p | KPSS p | verdict |
|---|---|---|---|
| log | 0.1282 | 0.01 (true p-value < 0.01) | non_stationary |

## 5. Autocorrelation, ARCH effects and normality of monthly log changes

```json
ts_autocorrelation {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log_diff","max_lag":24}
```

n = 214. Significant ACF lags: [3]; PACF: [3]. autocorrelation remains (smallest Ljung-Box p = 0.0391 across lags 1-24).

ARCH-LM p = 0.994: No ARCH effects detected: a GARCH model is not warranted at this frequency; constant variance is adequate.

Jarque-Bera p = <1e-4, skewness 1.651, kurtosis 8.865.

## 6. Same diagnostics on DAILY log changes

```json
ts_autocorrelation {"series_id":"GVZCLS","end":"2026-04-30","transform":"log_diff","max_lag":24}
```

n = 4507. Significant ACF lags: [1,2,3,6,8,15,19]; PACF: [1,2,3,6,7,8,12,15]. autocorrelation remains (smallest Ljung-Box p = 7.72e-11 across lags 1-24).

ARCH-LM p = <1e-4: ARCH effects present (volatility clustering): a GARCH-type variance model is warranted.

Jarque-Bera p = <1e-4, skewness 0.8552, kurtosis 10.13.

## 7. ARIMA candidates on log(GVZ)

```json
ts_fit_arima {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log","forecast_horizon":5}
```

**d = 0**

| order | ΔAICc | BIC | residuals white noise | warnings |
|---|---|---|---|---|
| (1,0,0) | 0 | -218.525 | true |  |
| (1,0,1) | 0.136 | -215.095 | true | ma1 is not significant (p = 0.157). |
| (2,0,0) | 0.1974 | -215.034 | true | ar2 is not significant (p = 0.169). |

**d = 1**

| order | ΔAICc | BIC | residuals white noise | warnings |
|---|---|---|---|---|
| (1,1,1) | 0 | -211.441 | true | drift is not significant (p = 0.720). MA inverse root 0.959 is close to 1 and nearly cancels the differencing: the series is likely over-differenced (try d = 0 with an AR term). |
| (2,1,2) | 0.6253 | -204.298 | true | drift is not significant (p = 0.788). ar2 is not significant (p = 0.454). ma2 is not significant (p = 0.741). MA inverse root 0.902 is close to 1 and nearly cancels the differencing: the series is likely over-differenced (try d = 0 with an AR term). |
| (0,1,0) | 4.721 | -213.317 | true | drift is not significant (p = 0.928). |
| (0,1,1) | 6.531 | -208.199 | true | drift is not significant (p = 0.932). ma1 is not significant (p = 0.616). |
| (1,1,0) | 6.573 | -208.157 | true | drift is not significant (p = 0.931). ar1 is not significant (p = 0.650). |

Selected (1,0,0): mean = 2.9164 (se 0.08812, p <1e-4); ar1 = 0.89532 (se 0.03059, p <1e-4); sigma2 = 0.019512 (se 0.001882)

Forecast (level median [95%]): 2026-05 28.837 [21.931, 37.919]; 2026-06 27.524 [19.06, 39.747]; 2026-07 26.399 [17.207, 40.501]; 2026-08 25.431 [15.879, 40.728]; 2026-09 24.594 [14.876, 40.66]

Selected (1,1,1): drift = -0.00099191 (se 0.002772, p 0.72); ar1 = 0.84425 (se 0.06511, p <1e-4); ma1 = -0.95927 (se 0.03863, p <1e-4); sigma2 = 0.019676 (se 0.001903)

Forecast (level median [95%]): 2026-05 28.604 [21.728, 37.655]; 2026-06 27.182 [18.829, 39.239]; 2026-07 26.033 [16.998, 39.87]; 2026-08 25.097 [15.711, 40.089]; 2026-09 24.329 [14.756, 40.111]


## 8. The course project's model: ARIMA(2,1,2)

```json
ts_fit_arima {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log","orders":[[2,1,2]],"forecast_horizon":5}
```

**d = 1**

| order | ΔAICc | BIC | residuals white noise | warnings |
|---|---|---|---|---|
| (2,1,2) | 0 | -204.298 | true | drift is not significant (p = 0.788). ar2 is not significant (p = 0.454). ma2 is not significant (p = 0.741). MA inverse root 0.902 is close to 1 and nearly cancels the differencing: the series is likely over-differenced (try d = 0 with an AR term). |

Selected (2,1,2): drift = -0.0009903 (se 0.003674, p 0.788); ar1 = 1.0209 (se 0.3578, p 0.00433); ar2 = -0.24953 (se 0.3335, p 0.454); ma1 = -1.0365 (se 0.3617, p 0.00417); ma2 = 0.12109 (se 0.366, p 0.741); sigma2 = 0.019356 (se 0.001871)

Forecast (level median [95%]): 2026-05 27.702 [21.091, 36.387]; 2026-06 26.048 [17.766, 38.19]; 2026-07 25.024 [16.021, 39.087]; 2026-08 24.388 [15.02, 39.597]; 2026-09 23.988 [14.4, 39.96]


## 9. GARCH(1,1)-t on monthly log changes

```json
ts_fit_garch {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log_diff","mean":"constant"}
```

Pre-test ARCH-LM p = 0.994. mu = -0.017799 (se 0.00969); omega = 0.00070752 (se 0.0001403); alpha = 3.8535e-11 (se null); beta = 0.96075 (se null); nu = 4.5848 (se 1.265).

Persistence 0.9607, half-life 17.3 periods.

vs constant variance: LR p = 0.431, ΔBIC = 9.048. GARCH does not earn its two extra parameters (BIC and the LR test favor constant variance). Use constant variance; keep the Student-t if tails are fat.

Warnings: Standard errors for alpha and beta are not reported because alpha is on its lower bound; the others are conditional on them. alpha is at its lower bound (~0): no detectable ARCH effect, so beta is not identified. GARCH is not warranted.

Forecast (log_diff, mean ± sd): 2026-05 -0.017799 ± 0.1343; 2026-06 -0.017799 ± 0.1343; 2026-07 -0.017799 ± 0.1343; 2026-08 -0.017799 ± 0.1343; 2026-09 -0.017799 ± 0.1343; 2026-10 -0.017799 ± 0.1343

## 10. AR(1)-GARCH(1,1)-t on monthly log level

```json
ts_fit_garch {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log","mean":"ar","ar_lags":1}
```

Pre-test ARCH-LM p = 0.999. mu = 0.30202 (se 0.07118); ar1 = 0.88748 (se 0.02454); omega = 0.00096019 (se 0.0002133); alpha = 8.3195e-12 (se null); beta = 0.94639 (se null); nu = 3.8688 (se 0.9205).

Persistence 0.9464, half-life 12.6 periods.

vs constant variance: LR p = 0.461, ΔBIC = 9.185. GARCH does not earn its two extra parameters (BIC and the LR test favor constant variance). Use constant variance; keep the Student-t if tails are fat.

Warnings: Standard errors for alpha and beta are not reported because alpha is on its lower bound; the others are conditional on them. alpha is at its lower bound (~0): no detectable ARCH effect, so beta is not identified. GARCH is not warranted.

Forecast (level median [95%]): 2026-05 27.985 [21.54, 36.357]; 2026-06 26.018 [18.336, 36.92]; 2026-07 24.389 [16.248, 36.609]; 2026-08 23.029 [14.751, 35.953]; 2026-09 21.886 [13.623, 35.162]; 2026-10 20.919 [12.745, 34.336]

## 11. GARCH(1,1)-t on DAILY log changes

```json
ts_fit_garch {"series_id":"GVZCLS","end":"2026-04-30","transform":"log_diff","mean":"constant","forecast_horizon":0}
```

Pre-test ARCH-LM p = <1e-4. mu = -0.0034273 (se 0.0006445); omega = 0.00029168 (se 0.00008129); alpha = 0.14335 (se 0.02637); beta = 0.76176 (se 0.04843); nu = 4.5878 (se 0.3312).

Persistence 0.9051, half-life 6.95 periods.

vs constant variance: LR p = <1e-4, ΔBIC = -293.4. GARCH improves the fit: volatility clustering matters for this series.


## 12. Out-of-sample backtest vs random walk

```json
ts_backtest {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","transform":"log","initial_window":72,"horizons":[1,3,6],"models":[{"kind":"random_walk"},{"kind":"random_walk","drift":true},{"kind":"arima","order":[1,1,1]},{"kind":"arima","order":[2,1,2]},{"kind":"arima","order":[1,0,0]},{"kind":"ar_garch","ar_lags":1,"garch":false,"distribution":"t"},{"kind":"ar_garch","ar_lags":1,"garch":true,"distribution":"t"}]}
```

143 origins (2014-06-01 to 2026-04-01), expanding window, 6044 ms. Top 5% of months = 0.468 of RW squared error.

**h = 1** — No model beats RW significantly at the 5% level for h=1.

| model | rel RMSE | DM p | cov80 | cov95 | log score | calm rel | turbulent rel |
|---|---|---|---|---|---|---|---|
| RW | 1 | — | 0.881 | 0.944 | 0.5044 | 1 | 1 |
| RW+drift | 1.004 | 0.0687 | 0.881 | 0.944 | 0.4995 | 1.007 | 1.003 |
| ARIMA(1,1,1) | 1.018 | 0.183 | 0.839 | 0.944 | 0.4866 | 1.022 | 1.015 |
| ARIMA(2,1,2) | 1.026 | 0.324 | 0.839 | 0.937 | 0.4604 | 1.019 | 1.03 |
| ARIMA(1,0,0) | 0.9826 | 0.235 | 0.867 | 0.937 | 0.5219 | 0.9895 | 0.9782 |
| AR(1)-t | 0.9844 | 0.315 | 0.832 | 0.937 | 0.6486 | 0.9864 | 0.9831 |
| AR(1)-GARCH(1,1)-t | 0.985 | 0.335 | 0.818 | 0.93 | 0.6401 | 0.9876 | 0.9833 |

**h = 3** — No model beats RW significantly at the 5% level for h=3.

| model | rel RMSE | DM p | cov80 | cov95 | log score | calm rel | turbulent rel |
|---|---|---|---|---|---|---|---|
| RW | 1 | — | 0.851 | 0.965 | 0.009881 | 1 | 1 |
| RW+drift | 1.014 | 0.091 | 0.858 | 0.95 | -0.004184 | 1.023 | 1.009 |
| ARIMA(1,1,1) | 1.026 | 0.0485 | 0.865 | 0.95 | -0.008007 | 1.025 | 1.026 |
| ARIMA(2,1,2) | 1.079 | 0.188 | 0.809 | 0.915 | -0.1201 | 1.007 | 1.112 |
| ARIMA(1,0,0) | 0.9574 | 0.335 | 0.794 | 0.957 | 0.05819 | 0.9835 | 0.9443 |
| AR(1)-t | 0.9653 | 0.405 | 0.766 | 0.936 | 0.1029 | 0.9542 | 0.9707 |
| AR(1)-GARCH(1,1)-t | 0.9667 | 0.42 | 0.78 | 0.929 | 0.1061 | 0.957 | 0.9714 |

**h = 6** — No model beats RW significantly at the 5% level for h=6.

| model | rel RMSE | DM p | cov80 | cov95 | log score | calm rel | turbulent rel |
|---|---|---|---|---|---|---|---|
| RW | 1 | — | 0.906 | 0.971 | -0.1223 | 1 | 1 |
| RW+drift | 1.03 | 0.0823 | 0.899 | 0.957 | -0.1402 | 1.061 | 1.016 |
| ARIMA(1,1,1) | 1.042 | 0.0686 | 0.899 | 0.971 | -0.1601 | 1.068 | 1.03 |
| ARIMA(2,1,2) | 1.245 | 0.0793 | 0.768 | 0.899 | -0.3881 | 1.141 | 1.289 |
| ARIMA(1,0,0) | 0.9709 | 0.737 | 0.855 | 0.964 | -0.02439 | 1.043 | 0.936 |
| AR(1)-t | 0.9842 | 0.815 | 0.797 | 0.949 | 0.01608 | 0.9531 | 0.9981 |
| AR(1)-GARCH(1,1)-t | 0.9891 | 0.87 | 0.797 | 0.949 | 0.01166 | 0.9593 | 1.002 |


## 13. Latest data: AR(1)-t forecast from the most recent complete month

```json
ts_fit_garch {"series_id":"GVZCLS","frequency":"m","transform":"log","mean":"ar","ar_lags":1,"variance":"constant","distribution":"t","forecast_horizon":6}
```

Pre-test ARCH-LM p = 0.999. mu = 0.29477 (se 0.06733); ar1 = 0.89019 (se 0.02301); sigma2 = 0.019981 (se 0.00491); nu = 3.5703 (se 0.8769).

Forecast (level median [95%]): 2026-11 21.875 [16.647, 28.745]; 2026-12 20.933 [14.522, 30.173]; 2027-01 20.128 [13.16, 30.784]; 2027-02 19.438 [12.191, 30.993]; 2027-03 18.843 [11.463, 30.976]; 2027-04 18.329 [10.898, 30.83]

## 14. Realized values after the sample end

```json
fred_get_observations {"series_id":"GVZCLS","frequency":"m","start":"2026-05-01","max_points":12}
```

n = 6, 2026-05-01 to 2026-10-01; dropped missing = 4; partial periods dropped = ["2026-11-01"].

## 15. Last in-sample level

```json
fred_get_observations {"series_id":"GVZCLS","frequency":"m","end":"2026-04-30","max_points":1}
```

n = 215, 2008-06-01 to 2026-04-01; dropped missing = 165; partial periods dropped = [].

## Realized check: forecasts from the course sample vs what happened

Last in-sample month (2026-04) averaged 30.3786. Actual monthly averages: 2026-05 25.9535, 2026-06 27.3676, 2026-07 25.2091, 2026-08 25.7214, 2026-09 25.2652.

| model | 5 forecasts (level median) | log RMSE |
|---|---|---|
| RW (no change) | 30.38, 30.38, 30.38, 30.38, 30.38 | 0.1626 |
| ARIMA(2,1,2) (course model) | 27.70, 26.05, 25.02, 24.39, 23.99 | 0.0495 |
| ARIMA(1,1,1) | 28.60, 27.18, 26.03, 25.10, 24.33 | 0.0501 |
| AR(1) on log level | 28.84, 27.52, 26.40, 25.43, 24.59 | 0.0531 |
| AR(1)-GARCH(1,1)-t | 27.98, 26.02, 24.39, 23.03, 21.89 | 0.0918 |

One five-month episode is an anecdote, not evidence; the rolling-origin backtest above is the evidence.
