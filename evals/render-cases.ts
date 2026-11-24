/** Writes evals/CASES.md: every case, its tags, the expected behaviour and the exact prompt, for review. */
import { writeFileSync } from "node:fs";
import { CASES } from "./cases.js";

const lines = [
  "# Eval cases (tool-use flow)",
  "",
  `${CASES.length} cases. \`tags[0]\` is the group. Every case is graded on two binary checks, **right tools** and **right answer**; **pass** requires both.`,
  "Graders are in `evals/cases.ts`; scripted reference behaviour in `evals/reference.ts`. Regenerate this file with `npx tsx evals/render-cases.ts`.",
  "",
  "| id | tags | expected |",
  "| --- | --- | --- |",
  ...CASES.map((c) => `| \`${c.id}\` | ${c.tags.join(", ")} | ${c.expected.replace(/\|/g, "\\|")} |`),
  "",
];
for (const c of CASES) lines.push(`## ${c.id}`, "", "````text", c.prompt, "````", "");
writeFileSync(new URL("./CASES.md", import.meta.url), lines.join("\n"));
console.log(`wrote evals/CASES.md (${CASES.length} cases)`);
