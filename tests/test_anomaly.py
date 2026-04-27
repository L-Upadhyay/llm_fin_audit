"""Tests for the earnings anomaly detector."""

from src.classical.anomaly_detector import (
    detect_earnings_anomaly,
    search_worst_period,
    analyze_trend,
)


def test_stable_earnings_no_anomaly():
    # Tightly clustered EPS, no quarter beyond 2 std deviations.
    eps = [1.50, 1.52, 1.48, 1.51, 1.49, 1.50, 1.52, 1.50]
    result = detect_earnings_anomaly(eps)
    print(f"\nStable: {result['summary']}")
    assert result["severity"] == "none"
    assert result["anomalies"] == []
    assert result["worst_quarter"] is None

    assert analyze_trend(eps) == "stable"


def test_one_clear_anomaly():
    # 7 normal quarters around 1.50 plus one large negative outlier at 0.20.
    eps = [1.50, 1.55, 0.20, 1.45, 1.50, 1.52, 1.48, 1.51]
    result = detect_earnings_anomaly(eps)
    print(f"\nAnomaly: {result['summary']}")
    assert len(result["anomalies"]) >= 1
    assert result["worst_quarter"]["eps"] == 0.20
    assert result["severity"] in ("moderate", "severe")


def test_declining_trend():
    # Newest-first: recent four quarters near 0.95, older four near 1.95.
    eps = [0.90, 0.95, 1.00, 0.95, 1.90, 1.95, 2.00, 1.95]
    trend = analyze_trend(eps)
    print(f"\nDeclining trend: {trend}")
    assert trend == "declining"


def test_search_worst_period():
    eps = [2.0, 1.5, 1.8, 0.5, 1.9, 2.1, 1.4]
    worst = search_worst_period(eps)
    print(f"\nWorst quarter: {worst}")
    assert worst["eps"] == 0.5
    assert worst["index"] == 3


if __name__ == "__main__":
    test_stable_earnings_no_anomaly()
    test_one_clear_anomaly()
    test_declining_trend()
    test_search_worst_period()
