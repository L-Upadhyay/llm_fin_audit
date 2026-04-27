"""Tests for src/classical/comparator.py."""

import os

from src.classical.comparator import (
    compare_stocks,
    plot_comparison,
    rank_stocks,
)


def test_compare_stocks_shape():
    # One ticker keeps the test fast but still exercises every code path.
    results = compare_stocks(["AAPL"])
    assert len(results) == 1
    r = results[0]

    expected_keys = {
        "ticker", "ratios", "csp_verdict", "kb_verdict",
        "kb_triggered_rules", "anomaly_severity", "anomaly_summary",
    }
    assert expected_keys.issubset(r.keys())
    assert r["ticker"] == "AAPL"
    assert r["csp_verdict"] in ("PASS", "WARNING", "FAIL")
    assert r["kb_verdict"] in ("PASS", "WARNING", "FAIL")
    assert r["anomaly_severity"] in ("none", "moderate", "severe")


def test_rank_stocks_orders_by_health():
    # Synthetic results — no network needed — to lock down ranking math.
    fake = [
        {
            "ticker": "BAD",
            "ratios": {},
            "csp_verdict": "FAIL",
            "kb_verdict": "FAIL",
            "kb_triggered_rules": [],
            "anomaly_severity": "severe",
            "anomaly_summary": "",
        },
        {
            "ticker": "GOOD",
            "ratios": {},
            "csp_verdict": "PASS",
            "kb_verdict": "PASS",
            "kb_triggered_rules": [],
            "anomaly_severity": "none",
            "anomaly_summary": "",
        },
        {
            "ticker": "MID",
            "ratios": {},
            "csp_verdict": "WARNING",
            "kb_verdict": "WARNING",
            "kb_triggered_rules": [],
            "anomaly_severity": "moderate",
            "anomaly_summary": "",
        },
    ]
    ranked = rank_stocks(fake)
    assert [r["ticker"] for r in ranked] == ["GOOD", "MID", "BAD"]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["risk_score"] == 0
    assert ranked[1]["rank"] == 2
    assert ranked[1]["risk_score"] == 3
    assert ranked[2]["rank"] == 3
    assert ranked[2]["risk_score"] == 6


def test_plot_comparison_writes_file(tmp_path):
    fake = [
        {
            "ticker": "AAA",
            "ratios": {"debt_to_equity": 1.0, "current_ratio": 1.5},
            "csp_verdict": "PASS",
            "kb_verdict": "PASS",
            "kb_triggered_rules": [],
            "anomaly_severity": "none",
            "anomaly_summary": "",
        },
        {
            "ticker": "BBB",
            "ratios": {"debt_to_equity": 3.0, "current_ratio": 0.7},
            "csp_verdict": "FAIL",
            "kb_verdict": "FAIL",
            "kb_triggered_rules": [],
            "anomaly_severity": "severe",
            "anomaly_summary": "",
        },
    ]
    out = str(tmp_path / "comparison.png")
    result = plot_comparison(fake, output_path=out)
    assert result == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


if __name__ == "__main__":
    test_rank_stocks_orders_by_health()
    test_compare_stocks_shape()
    # plot test needs tmp_path — easier to run via pytest
