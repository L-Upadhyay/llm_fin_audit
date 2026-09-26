"""
Facts: everything the classical layer knows about one ticker, in one dict.

This is the only material the verified pipeline lets the LLM see, and the
ground truth the claim verifier checks against.
"""

from datetime import datetime, timezone

from src.classical.anomaly_detector import detect_earnings_anomaly
from src.classical.consistency import check_consistency
from src.classical.csp_solver import FinancialCSP
from src.classical.knowledge_base import run_compliance_check
from src.classical.thresholds import missing_required
from src.data.loader import get_earnings_history, get_financial_ratios, get_realtime_price


# Live-quote fields worth giving the model (drop error/timestamp noise).
REALTIME_FIELDS = (
    "current_price", "previous_close", "price_change", "price_change_percent",
    "day_low", "day_high", "fifty_two_week_low", "fifty_two_week_high",
    "volume", "market_cap", "dividend_yield", "beta", "next_earnings_date",
)


def facts_from_data(ticker, ratios, realtime, eps_history):
    """Run the classical layer over already-fetched data (no network)."""
    eps_values = list(eps_history.values())
    anomaly = detect_earnings_anomaly(eps_values)
    kb = run_compliance_check(ratios)
    return {
        "ticker": ticker.upper(),
        "ratios": {k: v for k, v in ratios.items() if k != "ticker"},
        "realtime": {k: (realtime or {}).get(k) for k in REALTIME_FIELDS},
        "quote_time": (realtime or {}).get("timestamp"),
        "csp_verdict": FinancialCSP().solve(ratios),
        "kb_verdict": kb["verdict"],
        "kb_triggered_rules": kb["triggered_rules"],
        "anomaly_severity": anomaly["severity"],
        "anomaly_summary": anomaly["summary"],
        "quarterly_eps": eps_history,
        "missing_required_ratios": missing_required(ratios),
        "data_quality_issues": [i["issue"] for i in check_consistency(ratios)],
    }


def build_facts(ticker):
    """Fetch live data for `ticker` and run the full classical layer."""
    ticker = ticker.upper()
    facts = facts_from_data(
        ticker,
        get_financial_ratios(ticker),
        get_realtime_price(ticker),
        get_earnings_history(ticker).get("quarterly_eps", {}),
    )
    facts["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return facts


def prompt_view(facts):
    """Compact copy of the facts for the prompt (rounded, no raw EPS dates)."""
    def rnd(v):
        return round(v, 4) if isinstance(v, float) else v

    view = {
        "ticker": facts["ticker"],
        "csp_verdict": facts["csp_verdict"],
        "compliance_verdict": facts["kb_verdict"],
        "compliance_rules_fired": facts["kb_triggered_rules"],
        "ratios": {k: rnd(v) for k, v in facts["ratios"].items()},
        "live_quote": {k: rnd(v) for k, v in facts["realtime"].items()},
        "earnings_anomalies": facts["anomaly_summary"],
    }
    # Only include caveats that exist; an empty list invites the model to
    # write "caveat: none" filler.
    if facts["missing_required_ratios"]:
        view["missing_required_ratios"] = facts["missing_required_ratios"]
    if facts["data_quality_issues"]:
        view["data_quality_issues"] = facts["data_quality_issues"]
    return view
