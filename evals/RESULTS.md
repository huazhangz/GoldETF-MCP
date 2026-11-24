# Tool-use eval results: Claude Opus 5.5, 2026-11-24

**Setup:**

- **Model and settings:** `claude-opus-5-5`, adaptive thinking, effort `medium`, server-side refusal fallback
  (never triggered: every response was served by the requested model).
- **Data:** FRED data through 2026-10 (fixture).
- **Run:** one rep per case.
- **Harness:** approved by the repository owner at sha `d3448b749bc2`.
- **Spend:** **$3.73** measured from recorded usage (`npx tsx evals/cost.ts`), plus about $0.07 for two API
  shape checks, against a $5 limit.

## Results

| Variant | Cases | Pass | Right tools | Right answer | Median tool calls | Median latency | Median cost/case |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (as shipped) | 22 | **22/22** | 22/22 | 22/22 | 2 | 10.8 s | $0.085 |
| v1 (+ skill in the system prompt) | 19 | **19/19** | 19/19 | 19/19 | 2 | 10.6 s | $0.095 |

v1 was run on 19 of the 22 cases to stay inside the budget. The three it skipped (`stl_houst`,
`quarterly_vix_2024`, `cpi_yoy_sep2026`) are pure data questions that baseline passed. On the 19 shared cases
both variants pass every case and make the same number of tool calls (35).

## Reading the result

- **The guardrails work in practice, not just with scripted agents.** Opus 5.5 declined GARCH for monthly GVZ
  ("the GARCH(1,1) persistence of 0.96 is a fitting artifact, not a real result"). It also called the
  unemployment/yield level correlation spurious, warned that the April 2020 shock dominates an ARIMA forecast,
  and reported the CPI result per horizon ("not significant at 1 month, significant at 3 months").
- **The eval is at its ceiling for this model.** With 100% on baseline it cannot show what the skill, the
  descriptions or the guardrail text each contribute. The skill added about 12% cost per case and no
  measurable accuracy on Opus 5.5. That is a finding, not a failure, but it means the next runs need either:
  - the ablations `v2` (minimal descriptions) and `v3` (guardrail text stripped), to see whether the tool
    outputs are what produce the correct behaviour;
  - cheaper or weaker models (Sonnet 5.5, Haiku 5.5), where the guardrails and the skill are more likely to
    matter;
  - harder cases.
- **Resolution.** With 22 cases × 1 rep the noise floor on a pass rate is about ±21 points. Treat these
  numbers as "no failures observed", not as a precise rate.

## Grader corrections (made after the run, re-graded offline)

On reading the two baseline failures, both were grader false negatives; the model's answers were correct:

| Case | Model's answer | Grader bug | Fix |
| --- | --- | --- | --- |
| `beat_rw_cpi` | "Somewhat, but only clearly at three months ahead... that gain is significant." | (1) identified the benchmark by its display label, and the agent had renamed the random walk "No-change"; (2) the phrase "that gain is significant" was not recognised as affirming | Identify the benchmark from the call (`models[benchmark].kind`); accept "<gain> is significant" |
| `efficiency_cpi_acf` | "Yes. ... clearly autocorrelated" | Its negation guard matched the quoted null hypothesis in 'Ljung-Box tests reject "no autocorrelation"' | Match only real negations ("are not autocorrelated", "no significant autocorrelation") |

Both fixes were checked against the scripted agents: the oracle still passes 22/22, and the null and naive
agents still fail 22/22. The saved transcripts were then re-graded with `evals/regrade.ts`. Before the fix,
baseline was 20/22 and v1 was 18/19; the previous grades are kept in each row's `meta.grade_before_regrade`.
Because the graders changed, the harness sha changed too, and the next paid run needs a fresh
`--approve-harness`.

## Excerpts

**`spurious_unrate_dgs10`.** The agent called `ts_compare_series` on levels, then on changes, then on changes
before 2020:

> **Levels:** The raw correlation is −0.12. Both series trend over long periods (they fail stationarity tests),
> which makes the effective sample only about 18 independent observations. Adjusted for that, p = 0.65, so
> there's no real relationship... **Monthly changes (the proper test):** The correlation is −0.07 (adjusted
> p = 0.14)... Granger test... it's statistical precedence, not causation.

**`outlier_unrate_2020`:**

> April's jump from 4.4% to 14.8% is an error 25 standard deviations in size. The model reads that jump as
> momentum and pushes it forward... Here I'd use a no-change or judgment-based forecast with much wider
> uncertainty bands, or a model that allows fat-tailed errors.

The full rows are in `.claude/hillclimb/tool-use/{baseline,v1}/results.jsonl`. Transcripts are written to
`traces/`, which is not committed.
