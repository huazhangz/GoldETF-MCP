/**
 * Grader validation (no model calls, no cost): pushes the scripted agents through the same
 * harness and graders as the real eval and checks they separate correct from incorrect behaviour.
 *
 *   NODE_USE_ENV_PROXY=1 npx tsx evals/validate-graders.ts
 *
 * Writes .claude/hillclimb/tool-use-validation/{baseline=oracle, v1=null, v2=naive}/ in the
 * report layout and exits non-zero if the oracle misses a case or the null agent passes one.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { CASES } from "./cases.js";
import { gradeEvalCase, runEvalCase } from "./harness.js";
import { NAIVE } from "./reference.js";

const FLOW = ".claude/hillclimb/tool-use-validation";
const AGENTS = [
  { dir: "baseline", agent: "oracle", label: "Oracle: scripted correct behaviour (must pass every case)" },
  { dir: "v1", agent: "null", label: "Null: no tools, answers \"I don't know.\" (must fail every case)" },
  { dir: "v2", agent: "naive", label: "Naive: plausible but wrong (ignores guardrails, over-claims, guesses numbers)" },
] as const;

mkdirSync(FLOW, { recursive: true });
writeFileSync(
  join(FLOW, "_state.json"),
  JSON.stringify(
    {
      flow: "tool-use-validation",
      metrics: [
        { id: "pass", label: "pass", kind: "binary" },
        { id: "tools_ok", label: "right tools", kind: "binary" },
        { id: "answer_ok", label: "right answer", kind: "binary" },
        { id: "no_shuttle", label: "no data copy", kind: "binary" },
      ],
      perf_fields: [{ id: "tool_calls", label: "tool calls" }, { id: "latency_s", label: "latency", unit: "s" }],
    },
    null,
    2,
  ),
);

const summary: Record<string, { pass: number; n: number; failed: string[] }> = {};
for (const a of AGENTS) {
  const vdir = join(FLOW, a.dir);
  mkdirSync(join(vdir, "traces"), { recursive: true });
  if (a.dir !== "baseline") writeFileSync(join(vdir, "change.md"), `${a.label}\n`);
  const rows: string[] = [];
  const s = (summary[a.agent] = { pass: 0, n: 0, failed: [] as string[] });
  for (const c of CASES) {
    const t0 = Date.now();
    const run = await runEvalCase({ id: c.id }, { variant: "baseline", agent: a.agent });
    const g = await gradeEvalCase({ id: c.id }, run);
    rows.push(
      JSON.stringify({
        prompt_id: c.id,
        rep: 0,
        prompt: c.prompt,
        tags: c.tags,
        meta: { expected: c.expected, agent: a.agent, naive_targets_case: a.agent === "naive" ? c.id in NAIVE : undefined },
        status: "ok",
        stop_reason: "end_turn",
        latency_s: (Date.now() - t0) / 1000,
        tool_calls: run.tool_calls,
        grade: g.grade,
        explanation: g.explanation,
      }),
    );
    writeFileSync(join(vdir, "traces", `${c.id}_rep0.json`), JSON.stringify(run.transcript, null, 2));
    s.n++;
    if (g.grade.pass) s.pass++;
    else s.failed.push(`${c.id} (${g.explanation.pass})`);
  }
  writeFileSync(join(vdir, "results.jsonl"), rows.join("\n") + "\n");
}

for (const a of AGENTS) {
  const s = summary[a.agent]!;
  console.log(`${a.agent.padEnd(6)} ${s.pass}/${s.n} pass`);
  if (a.agent !== "null") for (const f of s.failed) console.log(`   fail: ${f}`);
}
const oracle = summary.oracle!;
const nul = summary.null!;
const naiveMisses = summary.naive!.pass;
const problems = [
  oracle.pass < oracle.n && `oracle failed ${oracle.n - oracle.pass} case(s): a grader or label is wrong`,
  nul.pass > 0 && `null agent passed ${nul.pass} case(s): a grader is too lenient`,
  naiveMisses > 0 && `naive agent passed ${naiveMisses} case(s): those graders cannot see that mistake`,
].filter(Boolean);
console.log(problems.length ? `\nPROBLEMS:\n- ${problems.join("\n- ")}` : "\nGraders separate correct from incorrect behaviour on every case.");
process.exit(oracle.pass < oracle.n || nul.pass > 0 ? 1 : 0);
