import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { FredClient } from "./fred/client.js";
import { registerFredTools } from "./tools/fred.js";
import { registerAnalysisTools } from "./tools/analysis.js";

export const SERVER_NAME = "fred-timeseries";
export const SERVER_VERSION = "0.1.0";

export function createServer(client: FredClient): McpServer {
  const server = new McpServer(
    { name: SERVER_NAME, version: SERVER_VERSION },
    {
      instructions:
        "Tools for FRED economic data and time-series diagnostics. Typical flow: fred_search_series -> " +
        "fred_get_series_info -> ts_test_stationarity (compare transforms) -> ts_autocorrelation on the stationary " +
        "transform. Analysis tools fetch data themselves from a series_id; do not copy long value arrays between calls.",
    },
  );
  registerFredTools(server, client);
  registerAnalysisTools(server, client);
  return server;
}
