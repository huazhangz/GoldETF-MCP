/**
 * Token footprint of each MCP tool definition (what every client session pays in context), measured with
 * the free count_tokens endpoint.
 *
 *   ANTHROPIC_API_KEY=... npx tsx evals/tool-tokens.ts
 */
import Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createServer } from "../src/server.js";
import { fixtureClient } from "./fixture.js";

const model = "claude-opus-5-5";
const anthropic = new Anthropic(process.env.EVAL_ANTHROPIC_BASE_URL ? { baseURL: process.env.EVAL_ANTHROPIC_BASE_URL } : {});
const server = createServer(fixtureClient());
const mcp = new Client({ name: "tool-tokens", version: "1.0.0" });
const [a, b] = InMemoryTransport.createLinkedPair();
await Promise.all([server.connect(b), mcp.connect(a)]);
const { tools } = await mcp.listTools();
const toApi = (t: (typeof tools)[number]): Anthropic.Tool => {
  const { $schema: _s, ...schema } = t.inputSchema as Record<string, unknown>;
  return { name: t.name, description: t.description ?? "", input_schema: schema as Anthropic.Tool.InputSchema };
};
const msg = [{ role: "user" as const, content: "hi" }];
const base = (await anthropic.messages.countTokens({ model, messages: msg })).input_tokens;
let sum = 0;
for (const t of tools) {
  const n = (await anthropic.messages.countTokens({ model, tools: [toApi(t)], messages: msg })).input_tokens - base;
  sum += n;
  console.log(`${t.name.padEnd(24)} ${String(n).padStart(5)} tokens  (description ${(t.description ?? "").length} chars, ${Object.keys((t.inputSchema as any).properties ?? {}).length} params)`);
}
const all = (await anthropic.messages.countTokens({ model, tools: tools.map(toApi), messages: msg })).input_tokens - base;
console.log(`all tools together: ${all} tokens (sum of singles ${sum})`);
await mcp.close();
