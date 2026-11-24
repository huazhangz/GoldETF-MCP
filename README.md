# fred-timeseries-mcp

An [MCP](https://modelcontextprotocol.io) server that gives Claude (or any MCP client) access to
[FRED](https://fred.stlouisfed.org) economic data **and a complete time-series modeling workflow**. The
workflow covers STL decomposition, stationarity tests, ACF/PACF, ARCH effects, ARIMA and AR-GARCH-t estimation,
and rolling-origin backtests against a random walk.

Every statistic is implemented in TypeScript, so the only install requirement is `node`. The results are
checked against statsmodels / arch / scipy. Log-likelihoods match to 1e-7 or better at identical parameters, and STL, ADF,
KPSS and the diagnostics match to 1e-8 or better.

> "Pull GVZCLS as monthly averages, check whether a GARCH model is warranted, and tell me honestly whether any
> model beats a random walk."

For a worked example, see the **[GVZ case study](docs/case-study/README.md)**. It redoes a course project
entirely through these tools. It finds that the course's ARIMA(2,1,2) is over-differenced and forecasts worse
than a random walk, and that GARCH is not warranted at monthly frequency (it is at daily frequency). Nothing
beats the random walk significantly, because a few fat-tailed shocks dominate the loss.

## Tools

| Tool | What it does |
| --- | --- |
| `fred_search_series` | Keyword search to get series IDs (e.g. "gold volatility" finds `GVZCLS`). Needs a key. |
| `fred_get_series_info` | Metadata: units, native frequency, coverage, notes. Needs a key. |
| `fred_get_observations` | Data with local aggregation (daily to monthly etc.; partial periods dropped) and log/diff transforms. Returns summary stats and a capped number of points. |
| `ts_decompose_stl` | STL decomposition (port of the reference Fortran), trend/seasonal strength F_T / F_S, monthly seasonal effects, dated largest shocks |
| `ts_test_stationarity` | ADF + KPSS with a combined verdict; compares several transforms in one call |
| `ts_autocorrelation` | ACF/PACF with significant lags, Ljung-Box, **ARCH-LM** and squared-ACF (is GARCH warranted?), Jarque-Bera |
| `ts_fit_arima` | Fits and ranks ARIMA candidates by AICc *within* each d. Exact Kalman-filter MLE, Hessian SEs, residual diagnostics, warnings for insignificant terms, cancelling or near-unit roots and over-differencing; forecasts back-transformed to levels |
| `ts_fit_garch` | Zero/constant/AR mean, GARCH(1,1) or constant variance, normal or Student-t errors (arch conventions, robust SEs). Pre-tests for ARCH effects and compares against constant variance |
| `ts_backtest` | Rolling-origin evaluation of up to 8 models: relative RMSE, Diebold-Mariano (HLN) vs benchmark, 80/95% coverage, interval score, log score, calm-vs-turbulent split, shock concentration |

The analysis tools fetch data themselves from a `series_id` (with a 10-minute cache), so the model never copies
arrays between calls. They also accept inline `values`, for example model residuals.

**No API key needed for data:** without `FRED_API_KEY`, observations come from FRED's public CSV download, so
every `fred_get_observations` and `ts_*` call works. Only search and metadata need a free key.

## Setup

```bash
git clone https://github.com/huazhangz/GoldETF-MCP.git && cd GoldETF-MCP
npm install && npm run build
```

**Claude Code**

```bash
claude mcp add fred-timeseries -e FRED_API_KEY=your_key -- node /absolute/path/to/GoldETF-MCP/dist/index.js
```

**Claude Desktop** (`claude_desktop_config.json`) or **Cursor** (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "fred-timeseries": {
      "command": "node",
      "args": ["/absolute/path/to/GoldETF-MCP/dist/index.js"],
      "env": { "FRED_API_KEY": "your_key" }
    }
  }
}
```

The `env` block is optional. Get a free key at <https://fredaccount.stlouisfed.org/apikeys>.

## Development

```bash
npm test          # 93 tests: parity with statsmodels/arch/scipy + in-memory MCP end-to-end tests
npm run typecheck
npm run inspect   # MCP Inspector against the built server
```

To reproduce the case study (it fetches live data; behind a proxy, Node needs `NODE_USE_ENV_PROXY=1`):

```bash
npx tsx scripts/case_study.ts
```

To re-run the backtest independently with statsmodels/arch:

```bash
python scripts/verify_case_study.py
```

To regenerate the test fixtures:

```bash
python scripts/gen_reference.py
```

The two Python scripts need `pip install statsmodels arch`. `scripts/gen_reference.py` uses only seeded
synthetic data.

```
src/
  fred/client.ts        FRED REST + keyless CSV, typed errors, cache
  stats/                pure numerics (no I/O): adf, kpss, autocorrelation, diagnostics, stl,
                        arima (Kalman MLE), volatility (AR-GARCH-t), backtest, evaluation, aggregate
  tools/                MCP tool definitions: fred, analysis, modeling
test/                   parity fixtures + end-to-end tests through a real MCP client
scripts/                case study, cross-check, fixture generator
docs/                   DESIGN.md (tool-design notes), case-study/
```

Design notes on naming, descriptions, error handling, and tools that tell the model when *not* to use a method
are in [docs/DESIGN.md](docs/DESIGN.md).

## License

MIT. GVZ and other series are the property of their sources and are fetched from FRED at run time.
