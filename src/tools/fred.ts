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
        "Find FRED economic data series by keyword and get their IDs. Use this first whenever you do not already know " +
        'the exact series ID (e.g. "gold volatility" -> GVZCLS, "10 year treasury" -> DGS10). ' +
        "Results are ordered by popularity and include frequency, units and date coverage.",
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
        "Get metadata for one FRED series: title, units, native frequency, seasonal adjustment, date coverage, " +
        "last update and the source's notes. Use it to check what a series measures and which frequencies it supports " +
        "before fetching or analyzing it.",
      inputSchema: { series_id: seriesId },
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
        "Fetch observations for a FRED series, optionally aggregated to a lower frequency (e.g. daily -> monthly average; " +
        "partially observed periods are dropped) and transformed (log, differences). Works without an API key. Always " +
        "returns summary statistics; returns at most `max_points` of the most recent points to save context. To run " +
        "statistics or models, call the ts_* tools with the same series_id instead of copying values out of this result.",
      inputSchema: {
        series_id: seriesId,
        start: dateStr.optional(),
        end: dateStr.optional(),
        frequency: frequency.optional(),
        aggregation: aggregation.optional(),
        keep_incomplete_periods: seriesSourceShape.keep_incomplete_periods,
        transform: transform.default("none"),
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
