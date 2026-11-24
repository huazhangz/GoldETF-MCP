import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { FredClient } from "./fred/client.js";
import { registerFredTools } from "./tools/fred.js";
import { registerAnalysisTools } from "./tools/analysis.js";
import { registerModelingTools } from "./tools/modeling.js";
import { registerRelationshipTools } from "./tools/relationship.js";

export const SERVER_NAME = "fred-timeseries";
export const SERVER_VERSION = "0.4.0";

export function createServer(client: FredClient): McpServer {
  const server = new McpServer(
    { name: SERVER_NAME, version: SERVER_VERSION },
    {
      instructions:
        "FRED economic data plus a time-series modeling workflow. Typical flow: fred_search_series -> fred_get_series_info -> " +
        "ts_decompose_stl (trend/seasonality) -> ts_test_stationarity (compare transforms) -> ts_autocorrelation (orders, ARCH " +
        "effects, normality) -> ts_fit_arima and, only if ARCH effects are present, ts_fit_garch -> ts_backtest against a random " +
        "walk before claiming forecasting skill; ts_compare_series for relationships between two series (on changes, not trending " +
        "levels). Analysis tools fetch data themselves from a series_id; do not copy long value " +
        "arrays between calls. Requires FRED_API_KEY. This product uses the FRED® API but is not endorsed or certified by " +
        "the Federal Reserve Bank of St. Louis.",
    },
  );
  registerFredTools(server, client);
  registerAnalysisTools(server, client);
  registerModelingTools(server, client);
  registerRelationshipTools(server, client);
  return server;
}
