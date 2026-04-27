"""
Tests for the benchmark harness.

We only test the classical condition here — the LLM and hybrid conditions
require a running Ollama instance, which is out of scope for an automated
test suite.
"""

from src.evaluation.benchmark import (
    _detect_violations,
    _stance,
    run_classical_only,
)


def test_run_classical_only_aapl():
    result = run_classical_only("AAPL")

    expected_keys = {
        "ticker", "ratios",
        "csp_verdict", "kb_verdict", "kb_triggered_rules",
        "anomaly_severity", "anomaly_summary",
        "verdict",
    }
    assert expected_keys.issubset(result.keys())
    assert result["ticker"] == "AAPL"
    assert result["csp_verdict"] in ("PASS", "WARNING", "FAIL")
    assert result["kb_verdict"] in ("PASS", "WARNING", "FAIL")
    assert result["verdict"] in ("PASS", "WARNING", "FAIL")
    assert result["anomaly_severity"] in ("none", "moderate", "severe")


def test_violation_detection_heuristic():
    # Classical FAIL but LLM says "healthy" -> violation flagged.
    violations = _detect_violations("FAIL", "Apple looks healthy and strong.")
    assert violations, "should flag positive language despite FAIL"

    # Classical PASS but LLM says "critical" -> violation flagged.
    violations = _detect_violations("PASS", "There are critical risks here.")
    assert violations, "should flag negative language despite PASS"

    # Aligned response -> no violation.
    assert _detect_violations("PASS", "The company looks healthy.") == []
    assert _detect_violations("FAIL", "There are critical concerns.") == []

    # Empty response -> no violation flagged (treated as missing data).
    assert _detect_violations("FAIL", None) == []
    assert _detect_violations("FAIL", "") == []


def test_stance_helper():
    assert _stance("This company is healthy and stable.") == "Positive"
    assert _stance("There are critical risks and warning signs.") == "Negative"
    assert _stance("Stable revenue but some concerning trends.") == "Mixed"
    assert _stance("Q3 results came in.") == "Neutral"
    assert _stance(None) == "n/a"


if __name__ == "__main__":
    test_run_classical_only_aapl()
    test_violation_detection_heuristic()
    test_stance_helper()
