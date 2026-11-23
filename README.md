# fred-timeseries-mcp

An [MCP](https://modelcontextprotocol.io) server that gives Claude (or any MCP client) access to
[FRED](https://fred.stlouisfed.org) economic data **plus the diagnostics you need before you fit a model**:
ADF and KPSS stationarity tests, ACF/PACF, and Ljung-Box.

It grew out of a course project on modeling the CBOE Gold ETF Volatility Index (FRED series `GVZCLS`) with
ARIMA. That workflow becomes a conversation:

> "Pull GVZCLS as monthly averages since 2008, find a transform that makes it stationary, and tell me what the
> ACF/PACF suggest for ARIMA orders."

## Tools

| Tool | What it does |
| --- | --- |
| `fred_search_series` | Keyword search, so you get the series ID (e.g. "gold volatility" gives `GVZCLS`) |
| `fred_get_series_info` | Metadata: units, native frequency, coverage, notes |
| `fred_get_observations` | Data with FRED-side aggregation (daily to monthly avg, etc.) and log/diff transforms; summary stats plus a capped number of points |
| `ts_test_stationarity` | ADF + KPSS with a combined verdict; compares several transforms in one call |
| `ts_autocorrelation` | ACF, PACF, significant lags, and the Ljung-Box test; can take residuals inline with `fit_df` |

The analysis tools take a `series_id` and fetch the data themselves, or take inline `values` for data that
isn't on FRED (such as model residuals). Statistical results match `statsmodels` 0.15 to 8 decimal places; see
`test/stats/reference.test.ts`.

## Setup

1. Get a free FRED API key: <https://fredaccount.stlouisfed.org/apikeys>
2. Build:

   ```bash
   git clone https://github.com/huazhangz/GoldETF-MCP.git && cd GoldETF-MCP
   npm install && npm run build
   ```

3. Register the server with your client.

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

Without a key the server still starts. The FRED tools then return a setup message, and the analysis tools
still work on inline `values`.

## Development

```bash
npm test          # unit + statsmodels parity + in-memory MCP end-to-end tests
npm run typecheck
npm run inspect   # MCP Inspector against the built server
```

`scripts/gen_reference.py` regenerates the statsmodels reference fixture.

Design notes on tool naming, descriptions and error handling are in [docs/DESIGN.md](docs/DESIGN.md).

## License

MIT
