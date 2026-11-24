import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { FredClient } from "../fred/client.js";
import { summarize } from "../stats/transforms.js";
import { aggregation, dataNote, dateStr, frequency, loadSeries, ok, round, safe, seriesId, seriesSourceShape, transform } from "./shared.js";

const READ_ONLY_REMOTE = { readOnlyHint: true, openWorldHint: true, idempotentHint: true } as const;

export function registerFredTools(server: McpServer, client: FredClient) {
  server.registerTool(
    "fred_search_series",
    {
      title: "Search FRED series",
      description:
        "Find FRED series IDs by keyword (\"gold volatility\" -> GVZCLS, \"10 year treasury\" -> DGS10). Use first when the ID is unknown. Returns frequency, units and coverage.",
      inputSchema: {
        query: z.string().trim().min(2).max(200).describe('Keywords, e.g. "gold ETF volatility" or "unemployment rate".'),
        limit: z.number().int().min(1).max(50).default(10).describe("Max results (1-50). Default 10."),
      },
      annotations: READ_ONLY_REMOTE,
    },
    safe(async ({ query, limit }) => {
      const { count, series } = await client.searchSeries(query, limit);
      if (series.length === 0) {
        return ok({ total_matches: 0, results: [], hint: "No matches. Try broader or different keywords." });
      }
      return ok({
        total_matches: count,
        results: series.map((s) => ({
          id: s.id,
          title: s.title,
          frequency: s.frequency_short,
          units: s.units_short,
          seasonal_adjustment: s.seasonal_adjustment_short,
          coverage: `${s.observation_start} to ${s.observation_end}`,
          popularity: s.popularity,
        })),
      });
    }),
  );

  server.registerTool(
    "fred_get_series_info",
    {
      title: "Get FRED series metadata",
      description:
        "Metadata for one series: units, native frequency, seasonal adjustment, coverage, notes, and whether it is third-party copyrighted.",
      inputSchema: { series_id: seriesId() },
      annotations: READ_ONLY_REMOTE,
    },
    safe(async ({ series_id }) => {
      const s = await client.getSeries(series_id);
      const notes = s.notes?.replace(/\s+/g, " ").trim();
      return ok({
        id: s.id,
        title: s.title,
        units: s.units,
        frequency: s.frequency,
        seasonal_adjustment: s.seasonal_adjustment_short,
        coverage: `${s.observation_start} to ${s.observation_end}`,
        last_updated: s.last_updated,
        notes: notes && notes.length > 1200 ? `${notes.slice(0, 1200)}…` : notes,
        third_party_copyright: /copyright/i.test(notes ?? ""),
        ...(/copyright/i.test(notes ?? "")
          ? { usage_note: "This series is owned by a third party. Beyond personal use, its owner's permission is required (FRED API terms)." }
          : {}),
      });
    }),
  );

  server.registerTool(
    "fred_get_observations",
    {
      title: "Get FRED observations",
      description:
        "Fetch a series, optionally aggregated to a lower frequency (partial periods dropped unless keep_incomplete_periods) and transformed. Returns summary stats and up to max_points recent points. For statistics, call ts_* tools with the series_id instead of copying values.",
      inputSchema: {
        series_id: seriesId(),
        start: dateStr().optional(),
        end: dateStr().optional(),
        frequency: frequency().optional(),
        aggregation: aggregation().optional(),
        keep_incomplete_periods: z.boolean().optional().describe("Keep a partial first/last period, e.g. month-to-date (default: dropped)."),
        as_of: seriesSourceShape().as_of,
        seasonal_period: z.number().int().min(2).max(366).optional().describe("Lag for seasonal_* transforms. Default from frequency (12 monthly)."),
        transform: transform().default("none"),
        max_points: z
          .number()
          .int()
          .min(0)
          .max(2000)
          .default(120)
          .describe("Max number of points to return (the most recent are kept). 0 returns only the summary. Default 120."),
      },
      annotations: READ_ONLY_REMOTE,
    },
    safe(async (args) => {
      const s = await loadSeries(client, args, undefined, 1);
      const { points } = s;
      const shown = args.max_points === 0 ? [] : points.slice(-args.max_points);
      const stats = summarize(points.map((p) => p.value));
      return ok({
        ...dataNote(s),
        summary: Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, round(v)])),
        points_returned: shown.length,
        ...(shown.length < points.length && args.max_points > 0
          ? { note: `Showing the last ${shown.length} of ${points.length} points. Narrow start/end or raise max_points for more.` }
          : {}),
        points: shown.map((p) => [p.date, round(p.value)]),
      });
    }),
  );
}
