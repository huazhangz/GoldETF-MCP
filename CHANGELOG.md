# Changelog

## 0.4.0

- **New tool `ts_compare_series`:** correlation between two series with an autocorrelation-adjusted p-value
  (effective sample size), lead/lag cross-correlation, Granger causality in both directions, and an OLS slope
  with Newey-West errors. It warns when non-stationary levels make the result likely spurious.
- **ALFRED vintages:** `as_of` on every data tool returns values as published on a date, before revisions.
- **Tool definitions are 27% smaller** (11.8k to 8.6k tokens):
  - terser descriptions;
  - `format: "date"` instead of regex patterns;
  - rarely used options kept only on `fred_get_observations`;
  - seasonal period inferred from the frequency.
- **No `$ref` pointers in tool schemas.** Shared zod instances used to produce them, and some MCP clients
  don't resolve them.
- **FRED client:**
  - retries network errors, 429 and 5xx with backoff;
  - LRU-capped response cache.
- **`ts_backtest`:** raises `step` automatically to keep runtime bounded on long (e.g. daily) series.
- **Eval:** 22 cases, adding correlation, spurious correlation and an intraday out-of-scope request. Also
  added: an API shape check and a tool token-footprint script.

## 0.3.0

- **Tool-use eval:** 20 two-sided cases, deterministic graders, scripted-agent validation, and four
  variants (baseline, +skill, minimal descriptions, no guardrail text).
- **Claude Code plugin:** skill plus MCP server, with the FRED key prompted as a sensitive setting.
- **`ts_fit_arima`:** warns when the last observations are extreme shocks.
- **`ts_backtest`:** accepts growth rates (`diff`, `log_diff`).
- **FRED compliance:**
  - keyless CSV access is opt-in (`FRED_CSV_FALLBACK=1`);
  - third-party copyrighted series are flagged;
  - the FRED API notice is displayed.

## 0.2.0

- **New modeling tools:**
  - `ts_decompose_stl`;
  - `ts_fit_arima` (exact Kalman MLE);
  - `ts_fit_garch` (AR-GARCH-t, arch conventions);
  - `ts_backtest` (rolling origin, Diebold-Mariano, interval and log scores).
- **`ts_autocorrelation`:** reports ARCH-LM and Jarque-Bera.
- **Aggregation:** local, and drops partial periods.
- **GVZ case study:** with an independent statsmodels/arch cross-check.

## 0.1.0

- **Initial server:** FRED search, metadata and observations.
- **Diagnostics:** ADF/KPSS stationarity and ACF/PACF/Ljung-Box, checked against statsmodels.
