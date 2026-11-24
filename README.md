# fred-timeseries-mcp

An [MCP](https://modelcontextprotocol.io) server and Claude Code plugin that gives an AI agent
[FRED®](https://fred.stlouisfed.org) economic data **and a statistically honest time-series workflow**:

- STL decomposition
- stationarity tests
- ACF/PACF and ARCH effects
- ARIMA and AR-GARCH-t estimation
- rolling-origin backtests against a random walk

It ships with a skill that tells the agent when each method is warranted.

> This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
> Not investment advice.

**Why it exists:** an agent with a GARCH tool will fit GARCH, and an agent with a forecasting tool will report a
forecast. These tools return the evidence for and against each method along with the estimates. They decline
GARCH when there is no volatility clustering, flag over-differenced and outlier-dominated ARIMA fits, and only
call a model better than a random walk when a Diebold-Mariano test says so.

| Question | What the tools find (data through 2026-10) |
| --- | --- |
| Is monthly CPI inflation forecastable? | **Yes.** AR(1) beats the random walk out of sample (RMSE -18% at 1 month, -35% at 3 months, DM p < 0.05) |
| Can a model beat a random walk for the 10-year yield or gold volatility (GVZ)? | **No**, not significantly |
| Does monthly GVZ need GARCH? | **No** (ARCH-LM p = 0.99). Daily GVZ and VIX do (p < 1e-4) |
| Are housing starts seasonal? | **Strongly** (F_S = 0.77). Gold volatility is not (F_S = 0.02) |
| Do VIX changes lead gold volatility? | Correlated (r = 0.60), and VIX helps predict GVZ (Granger p = 0.04), not the reverse |
| Are unemployment and the 10-year yield correlated? | The levels look correlated (naive p = 0.015), but the result is **spurious**: adjusted p = 0.65, and changes are not significantly related |

Worked example: the **[GVZ case study](https://github.com/huazhangz/GoldETF-MCP/blob/HEAD/docs/case-study/README.md)** redoes a course forecasting project
entirely through these tools.

## Tools

| Tool | What it does |
| --- | --- |
| `fred_search_series` / `fred_get_series_info` | Find series IDs. Metadata includes native frequency, coverage, and whether the series is third-party copyrighted |
| `fred_get_observations` | Data with local aggregation (partial periods dropped unless requested) and log/diff/seasonal transforms; summary stats and a capped number of points |
| `ts_decompose_stl` | STL (port of the reference Fortran), F_T / F_S, monthly seasonal effects, dated largest shocks |
| `ts_test_stationarity` | ADF + KPSS with a combined verdict; compares several transforms in one call |
| `ts_autocorrelation` | ACF/PACF with significant lags, Ljung-Box, ARCH-LM and squared ACF, Jarque-Bera |
| `ts_fit_arima` | Exact-MLE ARIMA candidates ranked by AICc *within* each d. Warns on insignificant terms, cancelling roots, over-differencing and extreme recent shocks. Forecasts are back-transformed to levels |
| `ts_fit_garch` | Zero/constant/AR mean, GARCH(1,1) or constant variance, normal or Student-t (arch conventions, robust SEs). Pre-tests for ARCH effects and compares against constant variance |
| `ts_backtest` | Rolling-origin evaluation of up to 8 models on levels or growth rates. Reports relative RMSE, Diebold-Mariano (HLN) tests, 80/95% coverage, interval and log scores, and a calm/turbulent split. Runtime is bounded automatically |
| `ts_compare_series` | Correlation with an autocorrelation-adjusted p-value, lead/lag CCF, two-way Granger causality, and an OLS slope with HAC errors, plus a spurious-correlation warning for non-stationary levels |

Every data tool accepts `as_of`, an ALFRED vintage date, to analyse data as published on that date, before
revisions. All 10 tool definitions together cost about 8.6k tokens of context.

All statistics are plain TypeScript, so installation only needs `node`. They are checked against statsmodels /
arch / scipy: log-likelihoods match to 1e-7 or better at identical parameters, and STL, ADF, KPSS, Granger, HAC
and the diagnostics match to 1e-8.

## Install

You need a free FRED API key: <https://fredaccount.stlouisfed.org/apikeys>.

**Claude Code plugin** (MCP server + skill; prompts for the key and stores it in the system keychain):

```
/plugin install fred-timeseries --marketplace huazhangz/GoldETF-MCP
```

**Any MCP client** (Claude Desktop `claude_desktop_config.json`, Cursor `.cursor/mcp.json`, ...):

```json
{
  "mcpServers": {
    "fred-timeseries": {
      "command": "npx",
      "args": ["-y", "fred-timeseries-mcp"],
      "env": { "FRED_API_KEY": "your_key" }
    }
  }
}
```

`FRED_CSV_FALLBACK=1` lets you read observations without a key from FRED's public CSV downloads. It is meant for
personal use only; search and metadata still need a key.

## Evaluation

`evals/` measures whether an agent *uses* the tools well, not just whether the statistics are right. The suite
has 22 cases, each graded on two deterministic checks: did the agent call the right tools, and is the answer
correct and consistent with what the tools returned? Coverage is two-sided: decline GARCH vs fit GARCH, "nothing
beats the random walk" vs "a model does", strong vs no seasonality, real vs spurious correlation, use tools vs answer without them.

Four variants separate what each layer contributes:

- `baseline`: as shipped
- `v1`: with the skill
- `v2`: minimal tool descriptions
- `v3`: guardrail text stripped from tool outputs

The graders were checked with scripted agents. A correct agent passes 22/22, while an agent that answers
"I don't know" and an agent that is plausible but wrong both pass 0/22. See [evals/README.md](https://github.com/huazhangz/GoldETF-MCP/blob/HEAD/evals/README.md).

## Development

```bash
npm test          # 108 tests: statsmodels/arch/scipy parity, MCP end-to-end, release metadata
npm run typecheck
```

```bash
npx tsx scripts/case_study.ts                                # GVZ case study
NODE_USE_ENV_PROXY=1 npx tsx evals/validate-graders.ts       # grader validation (no model calls)
```

```bash
python scripts/verify_case_study.py                          # independent statsmodels/arch re-run
python scripts/gen_reference.py                              # regenerate test fixtures
```

The two Python scripts need `pip install statsmodels arch`.

```
src/stats/   pure numerics: adf, kpss, autocorrelation, diagnostics, stl, arima (Kalman MLE),
             volatility (AR-GARCH-t), backtest, evaluation, relationship, aggregate
src/tools/   MCP tool definitions        src/fred/  FRED client (API + opt-in CSV)
plugin/      Claude Code plugin: manifest + skills/fred-timeseries/SKILL.md
evals/       tool-use eval: cases, graders, fixture, agent loop, runner
test/        parity fixtures + end-to-end tests      docs/  DESIGN.md, case-study/
```

Design notes (naming, descriptions, guardrails, error handling) are in [docs/DESIGN.md](https://github.com/huazhangz/GoldETF-MCP/blob/HEAD/docs/DESIGN.md).

## License and data

The code is MIT licensed. Data come from FRED at run time under the user's own API key and the
[FRED API terms](https://fred.stlouisfed.org/docs/api/terms_of_use.html). Some series are owned by third parties
(for example CBOE's VIX and GVZ). Beyond personal use, their owners' permission is required; `fred_get_series_info`
flags these series.
