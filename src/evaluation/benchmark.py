"""
Evaluation harness for llm_fin_audit.

Compares three experimental conditions on the same ticker + question:

    classical_only  — full classical layer (CSP + KB + anomaly detector)
    llm_only        — Agno agent on llama3.2 with NO tools (training knowledge only)
    hybrid          — FinancialAnalysisTeam with all classical tools

For each ticker we capture wall-clock response time and a heuristic
"constraint_violations" check — does the LLM contradict the classical
verdict in plain text? — then dump everything to JSON and a 2-panel
matplotlib chart.

The classical condition is the ground truth here: the CSP and KB are
deterministic and operate on real yfinance numbers, so any LLM-only
contradiction is treated as a violation.
"""

import json
import os
import time

import matplotlib
# Headless backend so the harness works in CI / SSH / no-display contexts.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agno.agent import Agent  # noqa: E402
from agno.models.ollama import Ollama  # noqa: E402

from src.classical.anomaly_detector import detect_earnings_anomaly  # noqa: E402
from src.classical.csp_solver import FinancialCSP  # noqa: E402
from src.classical.knowledge_base import run_compliance_check  # noqa: E402
from src.data.loader import get_earnings_history, get_financial_ratios  # noqa: E402
from src.llm.agno_agents import MODEL_ID, FinancialAnalysisTeam  # noqa: E402


DEFAULT_QUESTION = (
    "Is this company financially healthy? Consider its ratios, "
    "compliance posture, and earnings stability."
)


# ---------------------------------------------------------------------- #
# Three experimental conditions
# ---------------------------------------------------------------------- #

def run_classical_only(ticker):
    """
    Full classical layer — CSP + KB + anomaly detector — no LLM involved.

    Returns a dict with:
        ticker, ratios,
        csp_verdict, kb_verdict, kb_triggered_rules,
        anomaly_severity, anomaly_summary,
        verdict   (most-severe of csp_verdict and kb_verdict)
    """
    ratios = get_financial_ratios(ticker)
    csp_verdict = FinancialCSP().solve(ratios)
    kb = run_compliance_check(ratios)

    earnings = get_earnings_history(ticker)
    eps_values = list(earnings.get("quarterly_eps", {}).values())
    anomaly = detect_earnings_anomaly(eps_values)

    # Overall verdict = most severe of CSP and KB outcomes.
    severity = {"PASS": 0, "WARNING": 1, "FAIL": 2}
    overall = max(
        (csp_verdict, kb["verdict"]),
        key=lambda v: severity.get(v, 0),
    )

    return {
        "ticker": ticker.upper(),
        "ratios": ratios,
        "csp_verdict": csp_verdict,
        "kb_verdict": kb["verdict"],
        "kb_triggered_rules": kb["triggered_rules"],
        "anomaly_severity": anomaly["severity"],
        "anomaly_summary": anomaly["summary"],
        "verdict": overall,
    }


def run_llm_only(ticker, question):
    """
    LLM-only condition: an Agno agent with NO tools, asked to analyze the
    ticker entirely from its training-time knowledge.

    Requires Ollama running locally with the llama3.2 model pulled.
    """
    ticker = ticker.upper()
    agent = Agent(
        name="LLMOnlyAgent",
        model=Ollama(id=MODEL_ID),
        tools=[],  # explicit: no tools available
        instructions=[
            "You are a financial analyst.",
            f"Analyze the company with ticker '{ticker}' from your training knowledge.",
            "Do not say you cannot access data — give your best analysis.",
            "Mention concrete strengths or concerns where possible.",
        ],
        markdown=True,
    )
    prompt = f"Ticker: {ticker}\nQuestion: {question}"
    response = agent.run(prompt)
    return response.content


def run_hybrid(ticker, question):
    """
    Hybrid condition: full FinancialAnalysisTeam (DataAgent + AnalysisAgent
    + ComplianceAgent) with classical tools wired in.
    """
    team = FinancialAnalysisTeam(ticker=ticker)
    return team.run(ticker, question)


# ---------------------------------------------------------------------- #
# Heuristics for cross-condition comparison
# ---------------------------------------------------------------------- #

# Sentiment-style word lists used to extract a one-word stance from
# free-form LLM responses. Crude on purpose — the goal is to surface
# obvious contradictions, not to do real sentiment analysis.
_POSITIVE_WORDS = (
    "healthy", "strong", "safe", "stable", "solid", "no major concerns",
)
_NEGATIVE_WORDS = (
    "critical", "high risk", "concerning", "weak", "trouble",
    "warning", "risky", "should avoid",
)


def _stance(response):
    """One-word stance: Positive / Negative / Mixed / Neutral / n/a."""
    if not response:
        return "n/a"
    low = response.lower()
    has_pos = any(w in low for w in _POSITIVE_WORDS)
    has_neg = any(w in low for w in _NEGATIVE_WORDS)
    if has_pos and not has_neg:
        return "Positive"
    if has_neg and not has_pos:
        return "Negative"
    if has_pos and has_neg:
        return "Mixed"
    return "Neutral"


def _detect_violations(classical_verdict, llm_response):
    """
    Heuristic check: does the LLM response contradict the classical verdict?

    Returns a list of short human-readable violation strings (empty if none).
    Trips when the LLM uses positive language despite a classical FAIL,
    or negative language despite a classical PASS.
    """
    if not llm_response:
        return []
    low = llm_response.lower()
    violations = []

    if classical_verdict == "FAIL":
        for word in ("healthy", "strong", "safe", "no concerns", "low risk"):
            if word in low:
                violations.append(
                    f"LLM uses '{word}' despite classical verdict FAIL"
                )
                break
    elif classical_verdict == "PASS":
        for word in ("critical", "very risky", "should avoid", "fail"):
            if word in low:
                violations.append(
                    f"LLM uses '{word}' despite classical verdict PASS"
                )
                break

    return violations


# ---------------------------------------------------------------------- #
# Per-ticker evaluation
# ---------------------------------------------------------------------- #

def evaluate_ticker(ticker, question=DEFAULT_QUESTION):
    """
    Run all three conditions for `ticker` and bundle the results.

    LLM/Hybrid failures (e.g. Ollama not running) are caught so that the
    classical condition still produces a result — the failing condition
    just records its error.

    Returns a dict shaped like:
        {
          "ticker", "question",
          "classical": {...},
          "classical_verdict",
          "llm_response", "hybrid_response",
          "llm_stance", "hybrid_stance",
          "constraint_violations": [...],
          "response_times": {"classical", "llm_only", "hybrid"},
          "errors":         {"classical", "llm_only", "hybrid"},
        }
    """
    out = {"ticker": ticker.upper(), "question": question}

    # --- Classical -----------------------------------------------------
    t0 = time.perf_counter()
    out["classical"] = run_classical_only(ticker)
    classical_time = time.perf_counter() - t0
    out["classical_verdict"] = out["classical"]["verdict"]
    classical_error = None

    # --- LLM only ------------------------------------------------------
    t0 = time.perf_counter()
    try:
        out["llm_response"] = run_llm_only(ticker, question)
        llm_error = None
    except Exception as e:
        out["llm_response"] = None
        llm_error = str(e)
    llm_time = time.perf_counter() - t0

    # --- Hybrid --------------------------------------------------------
    t0 = time.perf_counter()
    try:
        out["hybrid_response"] = run_hybrid(ticker, question)
        hybrid_error = None
    except Exception as e:
        out["hybrid_response"] = None
        hybrid_error = str(e)
    hybrid_time = time.perf_counter() - t0

    out["response_times"] = {
        "classical": classical_time,
        "llm_only": llm_time,
        "hybrid": hybrid_time,
    }
    out["errors"] = {
        "classical": classical_error,
        "llm_only": llm_error,
        "hybrid": hybrid_error,
    }
    out["llm_stance"] = _stance(out.get("llm_response"))
    out["hybrid_stance"] = _stance(out.get("hybrid_response"))
    out["constraint_violations"] = _detect_violations(
        out["classical_verdict"], out.get("llm_response")
    )

    return out


# ---------------------------------------------------------------------- #
# Benchmark across multiple tickers
# ---------------------------------------------------------------------- #

def run_benchmark(tickers, question=DEFAULT_QUESTION,
                  output_path="data/benchmark_results.json"):
    """
    Evaluate every ticker, write the combined results to `output_path` as
    JSON, and return the same dict (keyed by uppercased ticker).
    """
    results = {}
    for t in tickers:
        results[t.upper()] = evaluate_ticker(t, question)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        # default=str rescues anything non-serializable (e.g. numpy floats).
        json.dump(results, f, indent=2, default=str)

    return results


def plot_benchmark_results(results, output_path="data/benchmark_chart.png"):
    """
    Render a 2-panel figure:
      - top:    response-time bar chart across the three conditions per ticker
      - bottom: verdicts table (classical verdict + LLM/Hybrid stance + violations)

    Saves to `output_path`. Returns the path.
    """
    if not results:
        return None

    tickers = list(results.keys())
    classical_times = [results[t]["response_times"]["classical"] for t in tickers]
    llm_times = [results[t]["response_times"]["llm_only"] for t in tickers]
    hybrid_times = [results[t]["response_times"]["hybrid"] for t in tickers]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # --- Top: response-time bar chart ---------------------------------
    x = np.arange(len(tickers))
    width = 0.27
    ax1.bar(x - width, classical_times, width, label="Classical", color="#1f77b4")
    ax1.bar(x,         llm_times,       width, label="LLM-only",  color="#ff7f0e")
    ax1.bar(x + width, hybrid_times,    width, label="Hybrid",    color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers)
    ax1.set_ylabel("Response time (s)")
    ax1.set_title("Response Time by Condition")
    ax1.legend()

    # --- Bottom: verdicts table ---------------------------------------
    ax2.axis("off")
    cell_text = []
    for t in tickers:
        r = results[t]
        violations = r.get("constraint_violations", [])
        cell_text.append([
            t,
            r.get("classical_verdict", "?"),
            r.get("llm_stance", "?"),
            r.get("hybrid_stance", "?"),
            str(len(violations)),
        ])
    table = ax2.table(
        cellText=cell_text,
        colLabels=["Ticker", "Classical", "LLM-only", "Hybrid", "Violations"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax2.set_title("Verdicts and Constraint Violations", pad=20)

    fig.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    # Local benchmark run. Requires Ollama for the LLM/hybrid conditions.
    results = run_benchmark(["AAPL", "MSFT"])
    plot_benchmark_results(results)
    print("Wrote data/benchmark_results.json and data/benchmark_chart.png")
