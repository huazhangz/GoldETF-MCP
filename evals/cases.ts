/**
 * Tool-use eval cases for the fred-timeseries MCP server.
 *
 * Each case grades two things separately:
 *   tools_ok  - did the agent call the right tool(s) with sensible arguments?
 *   answer_ok - is the final answer correct AND consistent with what the tools returned?
 *
 * Graders are deterministic (regexes + numeric checks against tool outputs or an independent data
 * oracle). Where the right answer depends on the agent's own choices (e.g. backtest settings), the
 * grader checks consistency with the agent's own tool output instead of a fixed label.
 *
 * Coverage is two-sided on purpose: "should decline GARCH" is paired with "should fit GARCH",
 * "nothing beats the random walk" with "a model does beat it", "no seasonality" with "strong
 * seasonality", and "use the tools" with "answer without tools".
 */
import { aggregate } from "../src/stats/aggregate.js";
import { csvRows } from "./fixture.js";

export interface ToolCall {
  name: string;
  args: Record<string, any>;
  ok: boolean;
  text: string;
  json?: any;
}
export interface AgentRun {
  answer: string;
  calls: ToolCall[];
}
export interface Check {
  tools_ok: boolean;
  answer_ok: boolean;
  notes: string[];
}
export interface EvalCase {
  id: string;
  tags: string[];
  prompt: string;
  /** What a correct answer looks like, for human review. */
  expected: string;
  grade: (run: AgentRun) => Promise<Check> | Check;
}

/* ------------------------------ helpers ------------------------------ */

const ok = (r: AgentRun, name?: string) => r.calls.filter((c) => c.ok && (!name || c.name === name));
const sid = (c: ToolCall) => String(c.args.series_id ?? "").toUpperCase();
const onSeries = (r: AgentRun, id: string, names: string[], pred: (c: ToolCall) => boolean = () => true) =>
  ok(r).filter((c) => names.includes(c.name) && sid(c) === id && pred(c));
const last = <T>(xs: T[]) => xs[xs.length - 1];
const any = (text: string, res: RegExp[]) => res.some((re) => re.test(text));
const nums = (text: string) =>
  [...text.replace(/(\d),(\d{3})/g, "$1$2").matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0])).filter(Number.isFinite);
const mentions = (text: string, v: number, tol: number) => nums(text).some((x) => Math.abs(x - v) <= tol);
const usesTransform = (c: ToolCall, ...ts: string[]) =>
  ts.includes(c.args.transform) || (Array.isArray(c.args.transforms) && c.args.transforms.some((t: string) => ts.includes(t)));
const monthly = (c: ToolCall) => c.args.frequency === "m";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** Independent data oracle: aggregates of the cached FRED data, computed without the MCP server. */
async function series(id: string) {
  return (await csvRows(id)).filter((r) => r.value !== null) as { date: string; value: number }[];
}

const NO_ARCH = [
  /no (statistically )?(significant |detectable |meaningful |evidence (of|for) )?(arch|garch|volatility clustering|conditional heteroskedasticity)/i,
  /not warranted|isn['’]t warranted|unwarranted|not justified|not supported by the data/i,
  /(alpha|α)[^.]{0,30}(essentially|effectively|approximately|practically|near|close to|≈|~) ?(zero|0)\b/i,
  /(alpha|α)[^.]{0,40}(boundary|lower bound)/i,
  /constant[- ]variance/i,
  /(beta|β)[^.]{0,40}(not identified|unidentified|meaningless)/i,
  /(does not|doesn['’]t|do not|don['’]t) (show|exhibit|display|have)[^.]{0,30}(clustering|arch)/i,
];
const YES_ARCH = [
  /(arch effects?|volatility clustering|conditional heteroskedasticity)[^.]{0,40}(present|significant|strong|evident|detected|clear|confirmed)/i,
  /(strong|significant|clear|pronounced|substantial) (evidence (of|for) )?(volatility clustering|arch effects?)/i,
  /(shows?|exhibits?|displays?|has) (strong |significant |clear |pronounced )?(volatility clustering|arch effects?)/i,
  /^\W*yes\b/i,
];
const DENY_BEAT = [
  /\b(not|no|n['’]t)\b[^.]{0,40}\b(significant|significantly|statistically)/i,
  /(cannot|can['’]t|does not|doesn['’]t|did not|didn['’]t|fails? to|failed to|unable to|could not|couldn['’]t)[^.]{0,30}(beat|outperform|improve on|do better)/i,
  /no model[^.]{0,40}(beat|outperform)/i,
  /^\W*no\b/i,
];
const AFFIRM_BEAT = [
  /(significantly|statistically significant(ly)?)[^.]{0,60}(better|beat|outperform|lower|improv)/i,
  /(gain|improvement|reduction|difference|advantage|edge|outperformance)[^.]{0,30}\b(is|was|are|were)\s+(statistically\s+)?significant/i,
  /(better|beat|outperform|lower|improv)[^.]{0,60}(statistically )?significant(ly)? at/i,
  /^\W*yes\b/i,
];

/** Shared rule for "does a model beat the random walk" questions: the answer must agree with the agent's own backtest. */
function backtestConsistency(r: AgentRun, id: string): Check {
  const bts = onSeries(r, id, ["ts_backtest"]);
  const bt = last(bts)?.json;
  const notes: string[] = [];
  if (!bt) return { tools_ok: false, answer_ok: false, notes: ["no successful ts_backtest call on " + id] };
  const anySig = bt.results.some((h: any) => h.summary.significantly_better_than_benchmark.length > 0);
  notes.push(`backtest says significant improvement: ${anySig}`);
  const affirm = any(r.answer, AFFIRM_BEAT);
  const deny = any(r.answer, DENY_BEAT);
  const answer_ok = anySig ? affirm : deny && !/^\W*yes\b/i.test(r.answer);
  if (!answer_ok) notes.push(anySig ? "answer does not report the significant improvement" : "answer does not state that no model is significantly better");
  // The benchmark is identified from the call, not its display label (agents often relabel models).
  const call = last(bts)!;
  const specs = Array.isArray(call.args.models) ? call.args.models : [];
  const vsRandomWalk = specs.length >= 2 && specs[call.args.benchmark ?? 0]?.kind === "random_walk";
  if (!vsRandomWalk) notes.push("backtest does not compare against a random-walk benchmark");
  return { tools_ok: vsRandomWalk, answer_ok, notes };
}

/* ------------------------------ cases ------------------------------ */

export const CASES: EvalCase[] = [
  {
    id: "garch_gvz_monthly",
    tags: ["garch", "should_decline", "guardrail"],
    prompt:
      "I want to model volatility clustering in the gold ETF volatility index (FRED series GVZCLS). Using monthly averages through October 2026, fit a GARCH(1,1) and tell me how persistent its volatility is.",
    expected: "Fits or tests, then says GARCH is not warranted at monthly frequency (ARCH-LM p ~0.99, alpha ~0); does not report persistence as a finding.",
    grade: (r) => {
      const t = onSeries(r, "GVZCLS", ["ts_fit_garch", "ts_autocorrelation"], monthly);
      return { tools_ok: t.length > 0, answer_ok: any(r.answer, NO_ARCH), notes: [] };
    },
  },
  {
    id: "garch_vix_daily",
    tags: ["garch", "should_fit", "guardrail"],
    prompt:
      "Do daily log changes of the VIX (FRED: VIXCLS) show volatility clustering? Use data from 2010 through October 2026. If they do, fit an appropriate model and report its persistence.",
    expected: "Daily ARCH-LM rejects; fits GARCH(1,1) on daily log changes and reports persistence (alpha + beta) matching the tool output.",
    grade: (r) => {
      const fits = onSeries(r, "VIXCLS", ["ts_fit_garch"], (c) => !c.args.frequency);
      const g = last(fits)?.json;
      const notes: string[] = [];
      const affirms = any(r.answer, YES_ARCH) && !/no (significant )?(volatility clustering|arch)/i.test(r.answer);
      let persistenceOk = false;
      if (g?.volatility?.persistence !== undefined) {
        const alpha = g.coefficients.find((c: any) => c.name === "alpha")?.estimate;
        const beta = g.coefficients.find((c: any) => c.name === "beta")?.estimate;
        persistenceOk = mentions(r.answer, g.volatility.persistence, 0.011) || (mentions(r.answer, alpha, 0.011) && mentions(r.answer, beta, 0.011));
        if (!persistenceOk) notes.push(`persistence ${g.volatility.persistence} not reported`);
      } else notes.push("no daily GARCH fit");
      return { tools_ok: fits.length > 0, answer_ok: affirms && persistenceOk, notes };
    },
  },
  {
    id: "beat_rw_gvz",
    tags: ["forecast_claims", "negative", "guardrail"],
    prompt:
      "Can an ARIMA-type model beat a random walk at forecasting the monthly average of GVZCLS one and three months ahead? Use data through October 2026 and give me a yes/no answer with evidence.",
    expected: "Runs ts_backtest; answers no: no model is significantly better than the random walk.",
    grade: (r) => backtestConsistency(r, "GVZCLS"),
  },
  {
    id: "beat_rw_dgs10",
    tags: ["forecast_claims", "negative", "guardrail"],
    prompt:
      "For monthly average 10-year Treasury yields (DGS10, 1990 through October 2026), does any ARIMA model forecast better than a random walk one month ahead? Test it.",
    expected: "Runs ts_backtest; no model is significantly better than the random walk.",
    grade: (r) => backtestConsistency(r, "DGS10"),
  },
  {
    id: "beat_rw_cpi",
    tags: ["forecast_claims", "positive", "guardrail"],
    prompt:
      "Is monthly US CPI inflation (the log change of CPIAUCSL) forecastable better than a naive no-change forecast? Use data from 1990 through October 2026 and test it out of sample one and three months ahead.",
    expected: "Runs ts_backtest on log_diff; an AR model beats the random walk significantly (DM p < 0.05) and the answer says so.",
    grade: (r) => {
      const c = backtestConsistency(r, "CPIAUCSL");
      const onGrowth = onSeries(r, "CPIAUCSL", ["ts_backtest"], (x) => x.args.transform === "log_diff").length > 0;
      if (!onGrowth) c.notes.push("backtest not run on log_diff (inflation)");
      return { ...c, tools_ok: c.tools_ok && onGrowth };
    },
  },
  {
    id: "outlier_unrate_2020",
    tags: ["outlier", "guardrail"],
    prompt: "Using monthly US unemployment rate data (UNRATE) through April 2020, fit an ARIMA model and forecast May 2020.",
    expected: "Fits ARIMA on data ending 2020-04 and warns that the April 2020 shock dominates the fit/forecast instead of presenting it uncritically.",
    grade: (r) => {
      const t = onSeries(r, "UNRATE", ["ts_fit_arima"], (c) => typeof c.args.end === "string" && c.args.end >= "2020-04-01" && c.args.end <= "2020-04-30");
      const caveat = any(r.answer, [/outlier|extreme|shock|pandemic|covid|unprecedented|dominat|unreliable|caution|structural break/i]);
      return { tools_ok: t.length > 0, answer_ok: caveat, notes: caveat ? [] : ["no caveat about the April 2020 shock"] };
    },
  },
  {
    id: "stl_houst",
    tags: ["seasonality", "positive"],
    prompt:
      "Are US housing starts (not seasonally adjusted, FRED: HOUSTNSA) seasonal? Quantify how strong the seasonality is and say which months are highest. Use data through October 2026.",
    expected: "ts_decompose_stl: strong seasonality (F_S ~0.77); names the peak months from the tool output (spring/summer).",
    grade: (r) => {
      const s = last(onSeries(r, "HOUSTNSA", ["ts_decompose_stl"]))?.json;
      if (!s) return { tools_ok: false, answer_ok: false, notes: ["no STL on HOUSTNSA"] };
      const fs = s.seasonal_strength;
      const strong = any(r.answer, [/(strong|pronounced|substantial|marked|clear|highly)[^.]{0,30}season/i]);
      const quantified = mentions(r.answer, fs, 0.021) || mentions(r.answer, fs * 100, 2.1);
      const top = [...s.seasonal_effects].sort((a: any, b: any) => b.effect - a.effect).slice(0, 3).map((e: any) => e.position);
      const month = top.some((m: string) => new RegExp(`\\b(${m}|${FULL[MONTHS.indexOf(m)]})\\b`, "i").test(r.answer));
      const notes = [!strong && "does not say strong", !quantified && `F_S ${fs} not reported`, !month && `none of top months ${top} named`].filter(Boolean) as string[];
      return { tools_ok: true, answer_ok: strong && quantified && month, notes };
    },
  },
  {
    id: "stl_gvz",
    tags: ["seasonality", "negative"],
    prompt: "Is there a meaningful month-of-year (seasonal) pattern in gold ETF volatility (GVZCLS, monthly averages through October 2026)?",
    expected: "ts_decompose_stl: seasonality is negligible (F_S ~0.02).",
    grade: (r) => {
      const t = onSeries(r, "GVZCLS", ["ts_decompose_stl"], monthly);
      const weak = any(r.answer, [
        /(negligible|weak|no (meaningful|significant|material|real|strong)|not (meaningful|significant|material|strong)|little|minimal|no evidence)[^.]{0,40}(season|pattern|month)/i,
        /(season\w*|pattern)[^.]{0,40}(negligible|weak|minimal|tiny|not meaningful)/i,
        /^\W*no\b/i,
      ]);
      return { tools_ok: t.length > 0, answer_ok: weak, notes: [] };
    },
  },
  {
    id: "stationarity_dgs10",
    tags: ["stationarity", "unit_root"],
    prompt:
      "Is the 10-year Treasury yield (DGS10, monthly averages through October 2026) stationary? What transformation would you apply before fitting an ARIMA model?",
    expected: "ts_test_stationarity on level and difference: level non-stationary, first difference stationary; recommends differencing.",
    grade: (r) => {
      const t = onSeries(r, "DGS10", ["ts_test_stationarity"]);
      const level = t.some((c) => usesTransform(c, "none", "log") || (!c.args.transform && !c.args.transforms));
      const diff = t.some((c) => usesTransform(c, "diff", "log_diff"));
      const answer_ok = any(r.answer, [/non-?stationary|not stationary|unit root/i]) && any(r.answer, [/differenc|d ?= ?1|log[_ ]diff|first diff|changes/i]);
      return { tools_ok: level && diff, answer_ok, notes: [] };
    },
  },
  {
    id: "stationarity_vix",
    tags: ["stationarity", "mean_reversion"],
    prompt: "Is the VIX (VIXCLS, monthly averages through October 2026) mean-reverting, or do I need to difference it before modeling?",
    expected: "ts_test_stationarity: the log level is stationary (ADF rejects, KPSS does not); mean-reverting, differencing not required.",
    grade: (r) => {
      const t = onSeries(r, "VIXCLS", ["ts_test_stationarity"]);
      const answer_ok =
        any(r.answer, [
          /mean[- ]revert/i,
          /(level|levels|log level|log-level|series)[^.]{0,30}(is|are) stationary/i,
          /stationary in (log )?levels?/i,
          /(no need|don['’]t need|do not need|need not|not necessary|unnecessary)[^.]{0,20}(to )?differenc/i,
        ]) && !/\b(you should|you need to|must) difference\b/i.test(r.answer);
      return { tools_ok: t.length > 0, answer_ok, notes: [] };
    },
  },
  {
    id: "quarterly_vix_2024",
    tags: ["data", "aggregation"],
    prompt: "What was the average VIX in each quarter of 2024?",
    expected: "Quarterly averages of daily VIXCLS for 2024Q1-Q4 (computed by the oracle from the same data), each within 0.06.",
    grade: async (r) => {
      const rows = (await series("VIXCLS")).filter((x) => x.date.startsWith("2024"));
      const q = aggregate(rows, "q", "avg").points.map((p) => p.value);
      const hit = q.map((v) => mentions(r.answer, Math.round(v * 100) / 100, 0.06));
      const t = ok(r).filter((c) => sid(c) === "VIXCLS");
      return { tools_ok: t.length > 0, answer_ok: hit.every(Boolean), notes: hit.every(Boolean) ? [] : [`expected ${q.map((v) => v.toFixed(2))}`] };
    },
  },
  {
    id: "partial_month_gvz",
    tags: ["data", "partial_period", "time_dependent"],
    prompt: "What has the average GVZ (GVZCLS) been so far in November 2026?",
    expected: "Mean of the daily November 2026 values available (partial month), labelled as month-to-date.",
    grade: async (r) => {
      const rows = (await series("GVZCLS")).filter((x) => x.date.startsWith("2026-11"));
      const mean = rows.reduce((s, x) => s + x.value, 0) / rows.length;
      const t = ok(r).filter((c) => sid(c) === "GVZCLS");
      const value = mentions(r.answer, mean, 0.1);
      const partial = any(r.answer, [/so far|to date|through|as of|partial|incomplete|month-to-date|trading days|november \d{1,2}|nov \d{1,2}|\d{1,2} nov/i]);
      return { tools_ok: t.length > 0, answer_ok: value && partial, notes: [`expected ${mean.toFixed(2)} over ${rows.length} days`] };
    },
  },
  {
    id: "cpi_yoy_sep2026",
    tags: ["data", "transform"],
    prompt: "What was year-over-year US CPI inflation (CPIAUCSL) in September 2026, in percent?",
    expected: "100 x (CPI 2026-09 / CPI 2025-09 - 1), or the log difference x 100, within 0.1 pp.",
    grade: async (r) => {
      const rows = await series("CPIAUCSL");
      const a = rows.find((x) => x.date === "2026-09-01")!.value;
      const b = rows.find((x) => x.date === "2025-09-01")!.value;
      const pct = (a / b - 1) * 100;
      const logp = Math.log(a / b) * 100;
      const t = ok(r).filter((c) => sid(c) === "CPIAUCSL");
      return { tools_ok: t.length > 0, answer_ok: mentions(r.answer, pct, 0.1) || mentions(r.answer, logp, 0.1), notes: [`expected ${pct.toFixed(2)}%`] };
    },
  },
  {
    id: "discover_gvz",
    tags: ["discovery"],
    prompt: "Which FRED series tracks the CBOE gold ETF volatility index? Give me its series ID, native frequency and the date its data start.",
    expected: "Searches; GVZCLS, daily, starting 2008-06-03.",
    grade: (r) => ({
      tools_ok: ok(r, "fred_search_series").length > 0,
      answer_ok: /GVZCLS/.test(r.answer) && /daily/i.test(r.answer) && /2008-06-03|june 3,? 2008|3 june 2008|06\/03\/2008/i.test(r.answer),
      notes: [],
    }),
  },
  {
    id: "wrong_id_recovery",
    tags: ["discovery", "error_recovery"],
    prompt: "Run ADF and KPSS stationarity tests on the FRED series GVZ using monthly averages.",
    expected: "The ID GVZ does not exist; the agent recovers (search -> GVZCLS) and runs the tests on GVZCLS.",
    grade: (r) => ({
      tools_ok: onSeries(r, "GVZCLS", ["ts_test_stationarity"]).length > 0,
      answer_ok: /GVZCLS/.test(r.answer),
      notes: [],
    }),
  },
  {
    id: "arima_choice_gvz",
    tags: ["modeling", "overdifferencing"],
    prompt: "Which ARIMA model would you use for monthly log GVZ (GVZCLS, data through April 2026), and why?",
    expected: "ts_fit_arima: prefers a d=0 mean-reverting model such as AR(1), or flags the d=1 models as over-differenced; does not pick ARIMA(2,1,2).",
    grade: (r) => {
      const t = onSeries(r, "GVZCLS", ["ts_fit_arima"], (c) => c.args.transform === "log");
      const d0 = /(ARIMA\s*\(\s*[0-3]\s*,\s*0\s*,\s*[0-3]\s*\)|\bAR\s*\(\s*[12]\s*\))/i.test(r.answer);
      const overdiff = /over-?differenc|mean[- ]revert/i.test(r.answer);
      const bad = /(recommend|choose|go with|select|prefer|pick|use)[^.]{0,40}ARIMA\s*\(\s*2\s*,\s*1\s*,\s*2\s*\)/i.test(r.answer);
      return { tools_ok: t.length > 0, answer_ok: (d0 || overdiff) && !bad, notes: bad ? ["recommends ARIMA(2,1,2)"] : [] };
    },
  },
  {
    id: "vix_forecast",
    tags: ["modeling", "faithfulness"],
    prompt: "Give me a 6-month forecast of the monthly average VIX (VIXCLS) with 95% prediction intervals, using data through October 2026.",
    expected: "A forecast tool on the log level (positive, right-skewed series); the numbers in the answer come from the tool's level forecasts.",
    grade: (r) => {
      const t = onSeries(r, "VIXCLS", ["ts_fit_arima", "ts_fit_garch"], (c) => c.args.transform === "log" && monthly(c));
      const res = last(t)?.json;
      const fcs = res ? (res.selected ? res.selected.flatMap((s: any) => s.forecast ?? []) : (res.forecast ?? [])) : [];
      const levels = fcs.flatMap((f: any) => (f.level ? [f.level.median, f.level.lo95, f.level.hi95] : []));
      const matched = levels.filter((v: number) => mentions(r.answer, v, 0.06)).length;
      return { tools_ok: t.length > 0, answer_ok: matched >= 6, notes: [`${matched} forecast numbers traced to tool output`] };
    },
  },
  {
    id: "correlation_gvz_vix",
    tags: ["relationship", "changes"],
    prompt: "What is the correlation between monthly changes in gold volatility (GVZCLS) and the VIX (VIXCLS) from 2010 through October 2026? Does either one lead the other?",
    expected: "ts_compare_series on monthly log changes: r ~0.60 (significant); VIX helps predict GVZ (Granger p ~0.04), not the reverse; strongest at lag 0.",
    grade: (r) => {
      const t = ok(r, "ts_compare_series").filter(
        (c) => [String(c.args.series_x), String(c.args.series_y)].sort().join() === "GVZCLS,VIXCLS" && (c.args.transform === "log_diff" || (c.args.transform_x === "log_diff" && c.args.transform_y === "log_diff")),
      );
      const res = last(t)?.json;
      if (!res) return { tools_ok: false, answer_ok: false, notes: ["no ts_compare_series on log changes of GVZCLS and VIXCLS"] };
      const traced = mentions(r.answer, res.correlation.pearson, 0.011);
      const leadLag = /lead|lag|granger|predict|precede/i.test(r.answer);
      return { tools_ok: true, answer_ok: traced && leadLag, notes: [!traced && `correlation ${res.correlation.pearson} not reported`, !leadLag && "lead/lag question not answered"].filter(Boolean) as string[] };
    },
  },
  {
    id: "spurious_unrate_dgs10",
    tags: ["relationship", "spurious", "guardrail"],
    prompt: "Is the 10-year Treasury yield (DGS10) correlated with the unemployment rate (UNRATE)? Use monthly data from 1990 through October 2026.",
    expected: "Level correlation looks significant (naive p ~0.015) but both levels are non-stationary: adjusted p ~0.65; in changes r ~-0.07, p ~0.14. Answer: no meaningful/significant relationship, with the spurious-levels caveat.",
    grade: (r) => {
      const t = ok(r, "ts_compare_series").filter((c) => [String(c.args.series_x), String(c.args.series_y)].sort().join() === "DGS10,UNRATE");
      const usedChanges = t.some((c) => ["diff", "log_diff"].includes(c.args.transform) || ["diff", "log_diff"].includes(c.args.transform_x));
      const caveat = /spurious|non-?stationar|trend|persistent|unit root|effective sample|in levels/i.test(r.answer);
      const deny = any(r.answer, [
        /(not|no|n['’]t)[^.]{0,40}(statistically )?significant/i,
        /no (meaningful|significant|reliable|robust|real) (correlation|relationship|link)/i,
        /(weak|negligible|essentially (zero|no))[^.]{0,30}(correlation|relationship)/i,
        /^\W*no\b/i,
      ]);
      const claims = /(significant(ly)?|strong(ly)?) (negative(ly)?|positive(ly)?|inverse(ly)?)? ?(correlat|related|relationship)/i.test(r.answer) && !deny;
      return {
        tools_ok: t.length > 0,
        answer_ok: (caveat || usedChanges) && deny && !claims,
        notes: [!(caveat || usedChanges) && "no spurious-levels caveat and no comparison of changes", !deny && "does not say the relationship is not significant"].filter(Boolean) as string[],
      };
    },
  },
  {
    id: "intraday_out_of_scope",
    tags: ["scope", "fabrication"],
    prompt: "What was the intraday high of the VIX on November 20, 2026?",
    expected: "FRED's VIXCLS has daily closes only: says the intraday high is not available (may give the close, labelled as such); does not invent a high.",
    grade: (r) => {
      const fabricated = /(intraday )?high[^.]{0,25}\b(was|of|=|at|reached|hit)\b\s*(about |around |roughly |approximately )?\$?\d/i.test(r.answer) && !/not (available|provided)|only .{0,20}clos/i.test(r.answer.split(/high/i)[0] ?? "");
      const ack = /clos(e|ing)/i.test(r.answer) && any(r.answer, [/not (available|provided|included)|isn['’]t (available|provided)|doesn['’]t (have|provide|include)|does not (have|provide|include)|only (provides|has|includes|publishes|reports|contains|gives)|no intraday|(cannot|can['’]t|unable to)[^.]{0,30}(give|provide|report|find)/i]);
      return { tools_ok: true, answer_ok: ack && !fabricated, notes: fabricated ? ["states an intraday high"] : ack ? [] : ["does not explain that only closes are available"] };
    },
  },
  {
    id: "concept_no_tools",
    tags: ["scope", "no_tool_needed"],
    prompt: "In one sentence each, what are the null hypotheses of the ADF and KPSS tests?",
    expected: "No tool calls; ADF H0 = unit root, KPSS H0 = stationarity.",
    grade: (r) => ({
      tools_ok: r.calls.length === 0,
      answer_ok: /unit root/i.test(r.answer) && /stationar/i.test(r.answer),
      notes: r.calls.length ? [`${r.calls.length} unnecessary tool call(s)`] : [],
    }),
  },
  {
    id: "efficiency_cpi_acf",
    tags: ["efficiency"],
    prompt: "Are monthly log changes in CPIAUCSL autocorrelated? Use data from 1990 through October 2026.",
    expected: "One ts_autocorrelation call by series_id (no copying of values); yes, strongly autocorrelated at short lags.",
    grade: (r) => {
      const byId = onSeries(r, "CPIAUCSL", ["ts_autocorrelation"], (c) => c.args.transform === "log_diff");
      const shuttled = r.calls.some((c) => Array.isArray(c.args.values) && c.args.values.length >= 50);
      const answer_ok =
        any(r.answer, [/^\W*yes\b/i, /significant(ly)?[^.]{0,30}autocorrelat/i, /(strong|strongly|positive|persistent)[^.]{0,30}autocorrelat/i, /are autocorrelated/i]) &&
        !/\b(are|is) not (significantly |meaningfully )?autocorrelated|\bno (significant|meaningful|evidence of) autocorrelation/i.test(r.answer);
      return { tools_ok: byId.length > 0 && !shuttled && r.calls.length <= 3, answer_ok, notes: [`${r.calls.length} tool calls${shuttled ? ", copied values" : ""}`] };
    },
  },
];
