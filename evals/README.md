# Tool-use eval

This eval asks whether an agent using this MCP server gets to the right answer for the right reasons. It
checks that the agent calls the right tools, respects the guardrails, and reports numbers that match what the
tools returned.

## What is measured

There are 20 cases in [CASES.md](CASES.md); the graders live in `cases.ts`. Each case is graded on two
deterministic checks, and **pass** requires both:

- `tools_ok`: did the agent call the right tools with sensible arguments? For example, `ts_backtest` before
  claiming a model beats a random walk, a log transform for VIX forecasts, and no tool call for a conceptual
  question.
- `answer_ok`: is the final answer correct **and consistent with the agent's own tool output**? Numbers are
  traced to tool results or to an independent data oracle. Where the right answer depends on the agent's own
  choices, such as backtest settings, the check is consistency with what its backtest returned, not a fixed
  label.

Also recorded: `no_shuttle` (the agent never copied 50+ values into a tool call), tool-call count, tokens,
latency and cost.

The cases are two-sided on purpose:

| Group | Should... | ...and the counterpart |
| --- | --- | --- |
| `garch` | decline GARCH for monthly GVZ (no ARCH effects) | fit GARCH for daily VIX (strong ARCH) |
| `forecast_claims` | say nothing beats the random walk (GVZ, 10y yield) | say AR beats it for CPI inflation (DM p < 0.05) |
| `seasonality` | find strong seasonality (housing starts, F_S 0.77) | find none (GVZ, F_S 0.02) |
| `stationarity` | difference the 10y yield | not difference VIX (mean-reverting) |
| `scope` | answer a concept question without tools | refuse to invent a correlation it cannot compute |

The remaining groups are `outlier` (April 2020 unemployment), `data` (quarterly means, month-to-date partial
period, YoY inflation), `discovery` (search, recovery from a wrong ID), `modeling` (over-differencing, faithful
forecast numbers) and `efficiency`.

Labels were derived with the validated tools (`ground-truth.ts`) on FRED data through 2026-10. One case,
`partial_month_gvz`, depends on the download date; its grader recomputes the answer from the same cached data.

## Variants

The server code is identical in all four variants; only what the model sees changes (`harness.ts`).

| Variant | What the model sees | Question it answers |
| --- | --- | --- |
| `baseline` | tool descriptions + server instructions as shipped | How good is the product today? |
| `v1` | baseline + `plugin/skills/fred-timeseries/SKILL.md` | Does the skill add value? |
| `v2` | tool title only, no parameter descriptions, no instructions | How much do the descriptions matter? |
| `v3` | baseline, with verdicts/warnings/guidance stripped from tool results | How much do the guardrail outputs matter? |

## Grader validation (done, no model calls)

```bash
NODE_USE_ENV_PROXY=1 npx tsx evals/validate-graders.ts
```

The script runs three scripted agents (`reference.ts`) through the same harness and graders. Results are in
`.claude/hillclimb/tool-use-validation/`:

| Agent | Pass |
| --- | --- |
| oracle (correct behaviour, answers built from real tool output) | **20/20** |
| null (no tools, "I don't know.") | **0/20** |
| naive (plausible but wrong: fits GARCH anyway, claims a lower RMSE "beats" the RW, guesses numbers, picks ARIMA(2,1,2), invents a correlation, ...) | **0/20** |

Every grader therefore accepts correct behaviour and rejects the specific mistake it targets. This does not yet
show that the graders accept *every* valid phrasing a real model might use. Review the first pilot's failures
by hand before trusting the headline.

## Running against Claude

The model calls use the Anthropic SDK and your API key; they cost money, so start with a pilot.

```bash
export ANTHROPIC_API_KEY=...
npx tsx evals/run-eval.mjs --flow .claude/hillclimb/tool-use --variant baseline --approve-harness   # once, after reviewing the harness
npx tsx evals/run-eval.mjs --flow .claude/hillclimb/tool-use --variant baseline --reps 1             # pilot
npx tsx evals/run-eval.mjs --flow .claude/hillclimb/tool-use --variant v1 --reps 3                   # etc.
node <claude-api skill>/shared/evals/report/build-report-lite.mjs .claude/hillclimb/tool-use/
```

- **Default model and settings:** `claude-opus-5-5` with adaptive thinking at `--effort medium`. Server-side
  refusal fallback is enabled; a response served by another model fails loudly as a served-model mismatch.
- **Changing the model:** pass `--model claude-sonnet-5-5`, for example. Claude Code users run various models,
  so the product should be measured on more than one.
- **First run:** the runner refuses to run until a human approves the harness files with `--approve-harness`.
  This is a change detector: an edited harness can't run unreviewed.
- **Data:** the FRED fixture serves real metadata from `fixtures/catalog.json` and real observations cached in
  `.cache/fred/` after the first download. That cache is gitignored, because third-party series such as CBOE's
  are not redistributed. No FRED key is needed.
- **Resolution:** with 20 cases × 3 reps, the noise floor on a pass rate is about ±13 points. Differences
  between variants smaller than that are not meaningful. Add reps or cases before acting on a small change.
