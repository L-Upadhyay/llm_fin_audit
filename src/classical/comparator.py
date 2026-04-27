"""
Multi-stock comparison module.

Pulls each ticker through the full classical layer (loader -> CSP -> KB ->
anomaly detector), produces side-by-side results, ranks them by overall
health score, and renders a matplotlib comparison chart with bars colored
by the CSP verdict.

Usage:
    results = compare_stocks(["AAPL", "MSFT", "TSLA"])
    ranked  = rank_stocks(results)
    path    = plot_comparison(ranked)
"""

import os

import matplotlib
# Use the non-interactive Agg backend so plotting works in headless contexts
# (CI, scripts run without a display, etc.).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after backend choice)

from src.classical.anomaly_detector import detect_earnings_anomaly
from src.classical.csp_solver import FinancialCSP
from src.classical.knowledge_base import run_compliance_check
from src.data.loader import get_earnings_history, get_financial_ratios


# Verdicts and severities mapped to numeric risk scores; lower = healthier.
VERDICT_SCORE = {"PASS": 0, "WARNING": 1, "FAIL": 2}
SEVERITY_SCORE = {"none": 0, "moderate": 1, "severe": 2}

# Bar colors keyed by CSP verdict (green / yellow / red).
VERDICT_COLOR = {"PASS": "#2ca02c", "WARNING": "#ffbf00", "FAIL": "#d62728"}


def compare_stocks(tickers):
    """
    Run the full classical pipeline for each ticker and collect the results
    side by side.

    Returns:
        list of dicts, one per ticker, each with:
            ticker, ratios, csp_verdict, kb_verdict, kb_triggered_rules,
            anomaly_severity, anomaly_summary
    """
    results = []
    for ticker in tickers:
        ratios = get_financial_ratios(ticker)

        # Constraint check
        csp_verdict = FinancialCSP().solve(ratios)

        # Compliance forward chaining
        kb = run_compliance_check(ratios)

        # Earnings anomaly detection
        earnings = get_earnings_history(ticker)
        eps_values = list(earnings.get("quarterly_eps", {}).values())
        anomaly = detect_earnings_anomaly(eps_values)

        results.append({
            "ticker": ticker.upper(),
            "ratios": ratios,
            "csp_verdict": csp_verdict,
            "kb_verdict": kb["verdict"],
            "kb_triggered_rules": kb["triggered_rules"],
            "anomaly_severity": anomaly["severity"],
            "anomaly_summary": anomaly["summary"],
        })
    return results


def rank_stocks(comparison_results):
    """
    Rank stocks from healthiest (rank 1) to riskiest by the sum of
    CSP-verdict score, KB-verdict score, and anomaly-severity score.

    Returns:
        list of dicts, each input augmented with `risk_score` and `rank`,
        sorted ascending by risk_score (lowest = healthiest).
    """
    scored = []
    for r in comparison_results:
        risk_score = (
            VERDICT_SCORE.get(r["csp_verdict"], 0)
            + VERDICT_SCORE.get(r["kb_verdict"], 0)
            + SEVERITY_SCORE.get(r["anomaly_severity"], 0)
        )
        scored.append({**r, "risk_score": risk_score})

    # Stable sort by score ascending — preserves input order on ties.
    scored.sort(key=lambda x: x["risk_score"])

    for i, r in enumerate(scored):
        r["rank"] = i + 1

    return scored


def plot_comparison(comparison_results, output_path="data/comparison_chart.png"):
    """
    Render two bar charts side by side — D/E and current ratio — with bars
    colored by each stock's CSP verdict (green=PASS, yellow=WARNING, red=FAIL).

    Threshold lines mark the CSP rule cutoffs (D/E > 2.0 and current_ratio
    < 1.0). Saves to `output_path` and returns the path.
    """
    if not comparison_results:
        return None

    tickers = [r["ticker"] for r in comparison_results]
    # Treat missing values as 0 so the bar still appears (annotated below).
    de_values = [r["ratios"].get("debt_to_equity") or 0 for r in comparison_results]
    cr_values = [r["ratios"].get("current_ratio") or 0 for r in comparison_results]
    colors = [VERDICT_COLOR.get(r["csp_verdict"], "gray") for r in comparison_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- D/E panel ----------------------------------------------------------
    axes[0].bar(tickers, de_values, color=colors)
    axes[0].set_title("Debt-to-Equity Ratio")
    axes[0].set_ylabel("D/E")
    axes[0].axhline(
        y=2.0, color="gray", linestyle="--", linewidth=0.8,
        label="warning threshold (2.0)",
    )
    axes[0].legend(fontsize=8, loc="upper right")
    for i, v in enumerate(de_values):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # --- Current-ratio panel -----------------------------------------------
    axes[1].bar(tickers, cr_values, color=colors)
    axes[1].set_title("Current Ratio")
    axes[1].set_ylabel("Current Ratio")
    axes[1].axhline(
        y=1.0, color="gray", linestyle="--", linewidth=0.8,
        label="critical threshold (1.0)",
    )
    axes[1].legend(fontsize=8, loc="upper right")
    for i, v in enumerate(cr_values):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Stock Comparison — bar color = CSP verdict "
        "(green=PASS, yellow=WARNING, red=FAIL)"
    )
    fig.tight_layout()

    # Make sure the output directory exists before writing.
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return output_path
