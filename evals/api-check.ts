/**
 * One-request API check (not an eval run): confirms the Anthropic API accepts the request shape the
 * eval harness sends (all MCP tool schemas, adaptive thinking, effort, prompt caching, server-side
 * fallback) and reports usage and the tool-definition token footprint. Costs well under $0.05.
 *
 *   ANTHROPIC_API_KEY=... npx tsx evals/api-check.ts [model]
 */
import Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createServer } from "../src/server.js";
import { fixtureClient } from "./fixture.js";

const model = process.argv[2] ?? "claude-opus-5-5";
const anthropic = new Anthropic(process.env.EVAL_ANTHROPIC_BASE_URL ? { baseURL: process.env.EVAL_ANTHROPIC_BASE_URL } : {});
const server = createServer(fixtureClient());
const mcp = new Client({ name: "api-check", version: "1.0.0" });
const [a, b] = InMemoryTransport.createLinkedPair();
await Promise.all([server.connect(b), mcp.connect(a)]);
const { tools: mcpTools } = await mcp.listTools();
const tools: Anthropic.Beta.BetaTool[] = mcpTools.map((t) => {
  const { $schema: _s, ...schema } = t.inputSchema as Record<string, unknown>;
  return { name: t.name, description: t.description ?? "", input_schema: schema as Anthropic.Beta.BetaTool.InputSchema };
});
const count = await anthropic.messages.countTokens({ model, tools: tools as Anthropic.Tool[], messages: [{ role: "user", content: "hi" }] });
const res = await anthropic.beta.messages.create({
  model,
  max_tokens: 2000,
  system: `Instructions from the MCP server "fred-timeseries":\n${mcp.getInstructions()}`,
  tools,
  messages: [{ role: "user", content: "Without calling any tool, name in one word the tool you would use to test whether a series is stationary." }],
  thinking: { type: "adaptive" },
  output_config: { effort: "low" },
  cache_control: { type: "ephemeral" },
  betas: ["server-side-fallback-2026-07-01"],
  fallbacks: "default",
});
console.log(
  JSON.stringify(
    {
      tools: tools.length,
      tool_definition_tokens: count.input_tokens,
      served_model: res.model,
      stop_reason: res.stop_reason,
      text: res.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join(" "),
      usage: res.usage,
    },
    null,
    1,
  ),
);
await mcp.close();
