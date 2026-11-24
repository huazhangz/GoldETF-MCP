/**
 * Eval harness: runs one case through an agent connected to the real MCP server (in-memory
 * transport, FRED fixture), and grades it.
 *
 * Agents:
 *   claude        - Claude via the Anthropic SDK (manual tool loop over the MCP tools)
 *   oracle|naive|null - scripted agents (evals/reference.ts) used to validate the graders
 *
 * Variants (what is being compared; the server code itself is identical in all of them):
 *   baseline - tool descriptions + server instructions as shipped
 *   v1       - baseline + the fred-timeseries skill (plugin/skills/fred-timeseries/SKILL.md) in the system prompt
 *   v2       - "minimal" server: tool title as the only description, no parameter descriptions, no instructions
 *   v3       - baseline, but guardrail text (verdicts, warnings, guidance) stripped from tool results
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createServer } from "../src/server.js";
import { CASES, type AgentRun, type EvalCase, type ToolCall } from "./cases.js";
import { fixtureClient } from "./fixture.js";
import { NAIVE, ORACLE } from "./reference.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const MAX_TURNS = 15;
const TODAY = "2026-11-24";

export interface Variant {
  descriptions: "full" | "minimal";
  instructions: boolean;
  skill: boolean;
  guardrailText: boolean;
}
export const VARIANTS: Record<string, Variant> = {
  baseline: { descriptions: "full", instructions: true, skill: false, guardrailText: true },
  v1: { descriptions: "full", instructions: true, skill: true, guardrailText: true },
  v2: { descriptions: "minimal", instructions: false, skill: false, guardrailText: true },
  v3: { descriptions: "full", instructions: true, skill: false, guardrailText: false },
};

/** Keys in tool results that carry interpretation rather than numbers. */
const GUARDRAIL_KEYS = new Set([
  "conclusion", "explanation", "verdict", "warnings", "guidance", "interpretation", "reading_guide", "note", "hint",
  "white_noise", "forecast_note", "significantly_better_than_benchmark", "significantly_worse_than_benchmark",
]);
function stripGuardrails(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(stripGuardrails);
  if (v && typeof v === "object") {
    return Object.fromEntries(Object.entries(v).filter(([k]) => !GUARDRAIL_KEYS.has(k)).map(([k, x]) => [k, stripGuardrails(x)]));
  }
  return v;
}
function stripDescriptions(schema: any): any {
  if (Array.isArray(schema)) return schema.map(stripDescriptions);
  if (schema && typeof schema === "object") {
    return Object.fromEntries(Object.entries(schema).filter(([k]) => k !== "description").map(([k, x]) => [k, stripDescriptions(x)]));
  }
  return schema;
}

export function loadEvalCases() {
  return CASES.map((c) => ({ id: c.id, prompt: c.prompt, tags: c.tags, meta: { expected: c.expected } }));
}

type Turn = { role: "system" | "user" | "assistant" | "tool_call" | "tool_result"; content: string; name?: string };

async function connect() {
  const server = createServer(fixtureClient());
  const mcp = new Client({ name: "eval", version: "1.0.0" });
  const [a, b] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(b), mcp.connect(a)]);
  return mcp;
}

async function execTool(mcp: Client, name: string, args: Record<string, any>, variant: Variant): Promise<ToolCall> {
  const r = (await mcp.callTool({ name, arguments: args })) as { isError?: boolean; content: { type: string; text: string }[] };
  const text = r.content.map((x) => x.text).join("\n");
  let json: unknown;
  if (!r.isError) {
    try {
      json = JSON.parse(text);
    } catch {
      /* non-JSON result */
    }
  }
  return { name, args, ok: !r.isError, text, json };
}
const resultForModel = (c: ToolCall, variant: Variant) =>
  variant.guardrailText || c.json === undefined ? c.text : JSON.stringify(stripGuardrails(c.json), null, 2);

/* ------------------------------ scripted agents ------------------------------ */

async function runScripted(kind: "oracle" | "naive" | "null", c: EvalCase, variant: Variant) {
  const mcp = await connect();
  const transcript: Turn[] = [{ role: "user", content: c.prompt }];
  const calls: ToolCall[] = [];
  let answer = "I don't know.";
  if (kind !== "null") {
    const script = (kind === "naive" ? NAIVE[c.id] : undefined) ?? ORACLE[c.id];
    if (!script) throw new Error(`no ${kind} script for ${c.id}`);
    for (const call of script.calls) {
      const res = await execTool(mcp, call.name, call.args, variant);
      calls.push(res);
      transcript.push({ role: "tool_call", name: call.name, content: JSON.stringify(call.args, null, 2) });
      transcript.push({ role: "tool_result", content: res.text });
    }
    answer = script.answer(calls.map((x) => x.json ?? null));
  }
  transcript.push({ role: "assistant", content: answer });
  await mcp.close();
  return { agentRun: { answer, calls } as AgentRun, transcript, model: undefined, usage: undefined, stop_reason: "end_turn" };
}

/* ------------------------------ Claude agent ------------------------------ */

function systemPrompt(variant: Variant, instructions: string | undefined) {
  const parts = [
    `You are a quantitative research assistant. Today's date is ${TODAY}. Use the available tools to answer the user's question with evidence. ` +
      "Keep the final answer under 200 words, include the key numbers, and say how you reached the conclusion.",
  ];
  if (variant.instructions && instructions) parts.push(`Instructions from the MCP server "fred-timeseries":\n${instructions}`);
  if (variant.skill) {
    const skill = readFileSync(join(HERE, "..", "plugin", "skills", "fred-timeseries", "SKILL.md"), "utf8").replace(/^---[\s\S]*?---\s*/, "");
    parts.push(`Skill "fred-timeseries" (loaded because the task involves FRED time-series analysis):\n${skill}`);
  }
  return parts.join("\n\n");
}

let anthropic: Anthropic | undefined;

async function runClaude(c: EvalCase, variant: Variant, model: string, effort: string) {
  anthropic ??= new Anthropic(process.env.EVAL_ANTHROPIC_BASE_URL ? { baseURL: process.env.EVAL_ANTHROPIC_BASE_URL } : {});
  const mcp = await connect();
  const { tools: mcpTools } = await mcp.listTools();
  const tools: Anthropic.Beta.BetaTool[] = mcpTools.map((t) => {
    const { $schema: _s, ...schema } = t.inputSchema as Record<string, unknown>;
    return {
      name: t.name,
      description: variant.descriptions === "minimal" ? (t.title ?? t.name) : (t.description ?? ""),
      input_schema: (variant.descriptions === "minimal" ? stripDescriptions(schema) : schema) as Anthropic.Beta.BetaTool.InputSchema,
    };
  });
  const system = systemPrompt(variant, mcp.getInstructions());
  const transcript: Turn[] = [{ role: "system", content: system }, { role: "user", content: c.prompt }];
  const messages: Anthropic.Beta.BetaMessageParam[] = [{ role: "user", content: c.prompt }];
  const calls: ToolCall[] = [];
  const usage = { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 };
  let servedModel: string | undefined;
  let stop_reason: string | null = null;
  let answer = "";
  let turns = 0;
  for (; turns < MAX_TURNS; turns++) {
    const res = await anthropic.beta.messages.create({
      model,
      max_tokens: 16000,
      system,
      tools,
      messages,
      thinking: { type: "adaptive" },
      output_config: { effort: effort as "low" | "medium" | "high" | "xhigh" | "max" },
      cache_control: { type: "ephemeral" },
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
    });
    servedModel = res.model;
    stop_reason = res.stop_reason;
    usage.input_tokens += res.usage.input_tokens;
    usage.output_tokens += res.usage.output_tokens;
    usage.cache_read_input_tokens += res.usage.cache_read_input_tokens ?? 0;
    usage.cache_creation_input_tokens += res.usage.cache_creation_input_tokens ?? 0;
    messages.push({ role: "assistant", content: res.content });
    const text = res.content.flatMap((b) => (b.type === "text" ? [b.text] : [])).join("\n").trim();
    if (res.stop_reason === "refusal") {
      answer = text;
      break;
    }
    if (res.stop_reason === "pause_turn") continue;
    const uses = res.content.filter((b): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use");
    if (text && uses.length) transcript.push({ role: "assistant", content: text });
    if (uses.length === 0) {
      answer = text;
      break;
    }
    const results: Anthropic.Beta.BetaToolResultBlockParam[] = [];
    for (const u of uses) {
      const call = await execTool(mcp, u.name, u.input as Record<string, any>, variant);
      calls.push(call);
      const content = resultForModel(call, variant);
      transcript.push({ role: "tool_call", name: u.name, content: JSON.stringify(u.input, null, 2) });
      transcript.push({ role: "tool_result", content });
      results.push({ type: "tool_result", tool_use_id: u.id, content, is_error: !call.ok });
    }
    messages.push({ role: "user", content: results });
  }
  transcript.push({ role: "assistant", content: answer || "(no final answer)" });
  await mcp.close();
  return {
    agentRun: { answer, calls } as AgentRun,
    transcript,
    model: servedModel,
    usage,
    stop_reason: turns >= MAX_TURNS ? "max_turns" : stop_reason,
    status: stop_reason === "max_tokens" ? "truncated" : "ok",
  };
}

/* ------------------------------ runner hooks ------------------------------ */

export async function runEvalCase(input: { id: string }, ctx: { variant: string; model?: string; agent?: string; effort?: string }) {
  const c = CASES.find((x) => x.id === input.id)!;
  const variant = VARIANTS[ctx.variant] ?? VARIANTS.baseline!;
  const agent = ctx.agent ?? "claude";
  const run =
    agent === "claude"
      ? await runClaude(c, variant, ctx.model ?? "claude-opus-5-5", ctx.effort ?? "medium")
      : await runScripted(agent as "oracle" | "naive" | "null", c, variant);
  return {
    ...run,
    output: run.agentRun.answer,
    tool_calls: run.agentRun.calls.length,
  };
}

export async function gradeEvalCase(input: { id: string }, run: Awaited<ReturnType<typeof runEvalCase>>) {
  const c = CASES.find((x) => x.id === input.id)!;
  const g = await c.grade(run.agentRun);
  const shuttled = run.agentRun.calls.some((x) => Array.isArray(x.args.values) && x.args.values.length >= 50);
  return {
    grade: {
      pass: g.tools_ok && g.answer_ok ? 1 : 0,
      tools_ok: g.tools_ok ? 1 : 0,
      answer_ok: g.answer_ok ? 1 : 0,
      no_shuttle: shuttled ? 0 : 1,
    },
    explanation: {
      pass:
        [!g.tools_ok && "tool use check failed", !g.answer_ok && "answer check failed", ...g.notes].filter(Boolean).join("; ") || "ok",
    },
  };
}
