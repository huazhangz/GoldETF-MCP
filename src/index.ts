#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { FredClient } from "./fred/client.js";
import { createServer } from "./server.js";

async function main() {
  const apiKey = process.env.FRED_API_KEY?.trim() || undefined;
  if (!apiKey) {
    // stdout is the MCP channel; diagnostics must go to stderr.
    console.error("[fred-timeseries] FRED_API_KEY not set: FRED tools will return an error until it is configured.");
  }
  const server = createServer(new FredClient({ apiKey }));
  await server.connect(new StdioServerTransport());
}

main().catch((err) => {
  console.error("[fred-timeseries] fatal:", err);
  process.exit(1);
});
