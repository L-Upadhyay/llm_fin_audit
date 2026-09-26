"""
Live evaluation: how often do real model answers state wrong numbers?

Runs the same questions over the facts snapshot under three conditions
and checks every answer with the claim verifier against the snapshot:

    llm_only   the model alone, no data (training knowledge only)
    team       the legacy Agno multi-agent team; its tools read the snapshot
    pipeline   the verified pipeline, recorded at three stages:
                 first attempt   (facts in context, before verification)
                 after rewrite   (one retry with the verifier's feedback)
                 final           (after in-place correction)

Every condition sees identical ground truth, so differences come from the
architecture, not data drift. Raw answers go to results/live_eval_raw.jsonl
(appended as it runs, so an interrupted run resumes where it stopped).

    python -m src.evaluation.live_eval                  # all tickers, llama3.2
    python -m src.evaluation.live_eval --provider ollama:qwen2.5:7b --tickers AAPL MSFT

Caveats
- The verifier only scores claims it can extract. Numbers in phrasings it
  misses count as "untracked", which is reported per condition.
- The delivered text is re-verified from the stored answer. Its error rate
  is near zero by construction (the same verifier found and corrected the
  errors), so the informative pipeline numbers are the first-attempt and
  after-rewrite rates, plus the verifier's recall on injected errors.
"""

import argparse
import json
import math
import os
import statistics
import time

from src.evaluation.snapshot import load_snapshot
from src.formatting import clean_agent_response, strip_comparison_block, strip_live_quote_block
from src.llm.pipeline import AuditPipeline
from src.llm.providers import get_provider
from src.verification.verifier import verify_text


RESULTS_DIR = "results"
QUESTIONS = (
    "Is {t} financially healthy? Cite the key ratios.",
    "What are the main balance-sheet risks for {t}? Include the relevant numbers.",
    "How profitable is {t}, and how is it valued?",
)

LLM_ONLY_SYSTEM = (
    "You are a financial analyst. Answer from your own knowledge. "
    "Cite specific ratio values where relevant. Be concise (under 180 words)."
)


# ---------------------------------------------------------------------- #
# Conditions
# ---------------------------------------------------------------------- #

def _answer_body(text):
    """The model-written part of a response (drop system-added blocks)."""
    text = strip_comparison_block(strip_live_quote_block(text or ""))
    return clean_agent_response(text.split("## Recommendation")[0]).strip()


def run_llm_only(provider, ticker, question):
    return provider.complete(LLM_ONLY_SYSTEM, f"Ticker: {ticker}\nQuestion: {question}")


def make_team(snapshot_facts):
    """Legacy Agno team whose data tools read the snapshot instead of yfinance."""
    import src.llm.agno_agents as agents

    def ratios(t):
        return {"ticker": t.upper(), **snapshot_facts[t.upper()]["ratios"]}

    def quote(t):
        f = snapshot_facts[t.upper()]
        return {"ticker": t.upper(), "timestamp": f.get("quote_time"), "error": None,
                **f["realtime"]}

    def eps(t, num_quarters=8):
        return {"ticker": t.upper(), "quarterly_eps": snapshot_facts[t.upper()]["quarterly_eps"]}

    agents.get_financial_ratios = ratios
    agents.get_realtime_price = quote
    agents.get_earnings_history = eps
    return agents.FinancialAnalysisTeam()


def score(text, facts, ticker):
    report = verify_text(text, {ticker: facts[ticker]}, ticker)
    s = report.summary()
    s["verdict_contradictions"] = sum(
        1 for c in report.contradicted if c.claim.kind == "verdict"
    )
    return s


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #

def _done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["condition"], r["ticker"], r["question"]) for r in map(json.loads, f)}


def run(provider_spec, tickers, raw_path):
    snapshot = load_snapshot()
    facts = snapshot["facts"]
    provider = get_provider(provider_spec)
    pipeline = AuditPipeline(provider=provider, fact_source=lambda t: facts[t.upper()])
    team = make_team(facts) if provider_spec.startswith("ollama:") else None
    team_model = provider_spec.split(":", 1)[1]
    if team is not None:
        team.model.id = team_model
    done = _done_keys(raw_path)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    with open(raw_path, "a") as out:
        for ticker in tickers:
            for q_template in QUESTIONS:
                question = q_template.format(t=ticker)
                for condition in ("llm_only", "team", "pipeline"):
                    if condition == "team" and team is None:
                        continue
                    if (condition, ticker, question) in done:
                        continue
                    row = {"condition": condition, "provider": provider_spec,
                           "ticker": ticker, "question": question}
                    t0 = time.perf_counter()
                    try:
                        if condition == "llm_only":
                            text = run_llm_only(provider, ticker, question)
                            row["answer"] = text
                            row["score"] = score(text, facts, ticker)
                        elif condition == "team":
                            text = _answer_body(team.run(ticker, question)["text"])
                            row["answer"] = text
                            row["score"] = score(text, facts, ticker)
                        else:
                            result = pipeline.run(ticker, question)
                            v = result["verification"]
                            row["answer"] = result["answer"]
                            row["score"] = v["final"]
                            row["first_attempt"] = v["first_attempt"]
                            row["retries"] = v["retries"]
                            row["corrections"] = v["corrections"]
                            row["first_answer"] = v["first_answer"]
                            row["uncorrected_answer"] = v["uncorrected_answer"]
                    except Exception as e:  # record and continue
                        row["error"] = f"{type(e).__name__}: {e}"
                    row["seconds"] = round(time.perf_counter() - t0, 2)
                    out.write(json.dumps(row, default=str) + "\n")
                    out.flush()
                    status = row.get("error") or row["score"]
                    print(f"{condition:9} {ticker:5} {row['seconds']:6.1f}s {status}", flush=True)


# ---------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------- #

def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _stage(rows, key):
    scores = [r[key] for r in rows if key in r]
    claims = sum(s["supported"] + s["contradicted"] for s in scores)
    wrong = sum(s["contradicted"] for s in scores)
    answers_wrong = sum(1 for s in scores if s["contradicted"] > 0)
    return {
        "answers": len(scores),
        "checkable_claims": claims,
        "claims_per_answer": round(claims / len(scores), 2) if scores else None,
        "claim_error_rate": round(wrong / claims, 4) if claims else None,
        "claim_error_ci95": wilson(wrong, claims),
        "answers_without_checkable_numbers": round(
            sum(1 for s in scores if s["supported"] + s["contradicted"] == 0) / len(scores), 4)
        if scores else None,
        "answers_with_error": round(answers_wrong / len(scores), 4) if scores else None,
        "answers_with_error_ci95": wilson(answers_wrong, len(scores)),
        "unverifiable_per_answer": round(statistics.mean(s["unverifiable"] for s in scores), 2)
        if scores else None,
        "untracked_numbers_per_answer": round(
            statistics.mean(s["untracked_numbers"] for s in scores), 2) if scores else None,
    }


def summarize(raw_path, provider_spec):
    facts = load_snapshot()["facts"]
    with open(raw_path) as f:
        rows = [r for r in map(json.loads, f) if r["provider"] == provider_spec]
    # Re-score every stored text with the current verifier so all
    # conditions and stages are judged by the same rules.
    for r in rows:
        if "answer" not in r:
            continue
        if r["condition"] == "pipeline":
            r["first"] = score(r["first_answer"], facts, r["ticker"])
            r["rewritten"] = score(r["uncorrected_answer"], facts, r["ticker"])
            # The delivered text (after in-place corrections) is re-verified
            # rather than assumed clean.
            r["delivered"] = score(r["answer"], facts, r["ticker"])
        else:
            r["score"] = score(r["answer"], facts, r["ticker"])
    out = {"provider": provider_spec, "conditions": {}}
    for cond in ("llm_only", "team", "pipeline"):
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        ok = [r for r in rs if "error" not in r]
        latency = sorted(r["seconds"] for r in ok)
        base = {
            "runs": len(rs),
            "errors": len(rs) - len(ok),
            "median_seconds": statistics.median(latency) if latency else None,
            "verdict_contradictions": sum(
                r.get("delivered", r.get("score", {})).get("verdict_contradictions", 0) for r in ok),
        }
        if cond == "pipeline":
            out["conditions"]["pipeline_first_attempt"] = {**base, **_stage(ok, "first")}
            out["conditions"]["pipeline_after_rewrite"] = {**base, **_stage(ok, "rewritten")}
            out["conditions"]["pipeline_final"] = {**base, **_stage(ok, "delivered")}
            out["conditions"]["pipeline_final"]["rewrites"] = sum(r["retries"] for r in ok)
            out["conditions"]["pipeline_final"]["corrections"] = sum(len(r["corrections"]) for r in ok)
        else:
            out["conditions"][cond] = {**base, **_stage(ok, "score")}
    return out


def to_markdown(s):
    names = {
        "llm_only": "LLM only (no data)",
        "team": "Agent team (tools, unverified)",
        "pipeline_first_attempt": "Pipeline — first attempt",
        "pipeline_after_rewrite": "Pipeline — after 1 rewrite",
        "pipeline_final": "Pipeline — delivered (after correction, re-verified)",
    }

    def pct(v):
        return "–" if v is None else f"{v:.1%}"

    def ci(v):
        return "" if not v else f" ({v[0]:.0%}–{v[1]:.0%})"

    lines = [
        f"# Live evaluation — {s['provider']}",
        "",
        "| Condition | Answers | Checkable claims / answer | Wrong claims | "
        "Answers with ≥1 wrong claim | Answers with no checkable numbers | "
        "Unsourced claims / answer | Median latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, c in s["conditions"].items():
        lines.append(
            f"| {names.get(key, key)} | {c['answers']} | {c['claims_per_answer']} | "
            f"{pct(c['claim_error_rate'])}{ci(c['claim_error_ci95'])} | "
            f"{pct(c['answers_with_error'])}{ci(c['answers_with_error_ci95'])} | "
            f"{pct(c['answers_without_checkable_numbers'])} | "
            f"{c['unverifiable_per_answer']} | {c['median_seconds']:.1f}s |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--provider", default="ollama:llama3.2")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--summarize-only", action="store_true",
                        help="re-score results/live_eval_raw.jsonl without calling the model")
    args = parser.parse_args()

    tickers = args.tickers or sorted(load_snapshot()["facts"])
    raw_path = os.path.join(RESULTS_DIR, "live_eval_raw.jsonl")
    if not args.summarize_only:
        run(args.provider, tickers, raw_path)

    summary = summarize(raw_path, args.provider)
    slug = args.provider.replace(":", "_").replace("/", "_")
    with open(os.path.join(RESULTS_DIR, f"live_eval_{slug}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = to_markdown(summary)
    with open(os.path.join(RESULTS_DIR, f"live_eval_{slug}.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
