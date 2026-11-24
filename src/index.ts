#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { FredClient } from "./fred/client.js";
import { createServer } from "./server.js";

async function main() {
  const apiKey = process.env.FRED_API_KEY?.trim() || undefined;
  const csvFallback = /^(1|true|yes)$/i.test(process.env.FRED_CSV_FALLBACK ?? "");
  if (!apiKey) {
    // stdout is the MCP channel; diagnostics must go to stderr.
    console.error(
      csvFallback
        ? "[fred-timeseries] no FRED_API_KEY: using public CSV downloads for observations (personal use); search/metadata disabled."
        : "[fred-timeseries] FRED_API_KEY not set: tools will return setup instructions until it is configured.",
    );
  }
  const server = createServer(new FredClient({ apiKey, csvFallback }));
  await server.connect(new StdioServerTransport());
}

main().catch((err) => {
  console.error("[fred-timeseries] fatal:", err);
  process.exit(1);
});
