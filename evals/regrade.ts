/**
 * Re-grade a finished run from its saved traces, without calling any model. Used when a grader bug is
 * fixed after a paid run: the agent's behaviour doesn't depend on the grader, so the transcripts can
 * be scored again. The previous grade is kept in meta.grade_before_regrade for transparency.
 *
 *   npx tsx evals/regrade.ts .claude/hillclimb/tool-use baseline v1
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { AgentRun, ToolCall } from "./cases.js";
import { gradeEvalCase } from "./harness.js";

type Turn = { role: string; content: string; name?: string };

/** Rebuild the agent run from a trace: successful tool results are JSON objects, errors are plain text. */
function fromTrace(turns: Turn[]): AgentRun {
  const calls: ToolCall[] = [];
  for (let i = 0; i < turns.length; i++) {
    const t = turns[i]!;
    if (t.role !== "tool_call") continue;
    const res = turns[i + 1]?.role === "tool_result" ? turns[i + 1]!.content : "";
    let json: unknown;
    try {
      json = JSON.parse(res);
    } catch {
      /* error text */
    }
    const ok = typeof json === "object" && json !== null;
    calls.push({ name: t.name!, args: JSON.parse(t.content), ok, text: res, json: ok ? json : undefined });
  }
  const final = turns.at(-1)!;
  const answer = final.role === "assistant" && final.content !== "(no final answer)" ? final.content : "";
  return { answer, calls };
}

const [flow, ...variants] = process.argv.slice(2);
if (!flow || variants.length === 0) throw new Error("usage: npx tsx evals/regrade.ts <flow> <variant>...");
for (const v of variants) {
  const path = join(flow, v, "results.jsonl");
  const rows = readFileSync(path, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  let changed = 0;
  for (const row of rows) {
    const turns: Turn[] = JSON.parse(readFileSync(join(flow, v, "traces", `${row.prompt_id}_rep${row.rep}.json`), "utf8"));
    const agentRun = fromTrace(turns);
    const g = await gradeEvalCase({ id: row.prompt_id }, { agentRun } as Parameters<typeof gradeEvalCase>[1]);
    if (JSON.stringify(g.grade) !== JSON.stringify(row.grade)) {
      changed++;
      row.meta = { ...(row.meta ?? {}), grade_before_regrade: row.grade };
      console.log(`${v} ${row.prompt_id}: pass ${row.grade.pass} -> ${g.grade.pass} (${g.explanation.pass})`);
    }
    row.grade = g.grade;
    row.explanation = g.explanation;
  }
  writeFileSync(path, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log(`${v}: ${rows.length} rows re-graded, ${changed} changed`);
}
