"""Tests for accounting-consistency constraints."""

from src.classical.consistency import check_consistency


def test_consistent_ratios_pass():
    ratios = {"quick_ratio": 1.1, "current_ratio": 1.35, "net_profit_margin": 0.39,
              "gross_margin": 0.68, "roe": 0.34, "debt_to_equity": 0.18, "pe_ratio": 24.7}
    assert check_consistency(ratios) == []


def test_quick_above_current_flags_period_mismatch():
    # AAPL in the May 2026 benchmark: annual current ratio vs TTM quick ratio.
    issues = check_consistency({"quick_ratio": 0.906, "current_ratio": 0.893})
    assert [i["ratios"] for i in issues] == [["quick_ratio", "current_ratio"]]
    assert "different periods" in issues[0]["issue"]


def test_small_rounding_differences_are_tolerated():
    assert check_consistency({"quick_ratio": 1.005, "current_ratio": 1.0}) == []


def test_sign_mismatch_between_roe_and_margin():
    issues = check_consistency({"roe": -0.15, "net_profit_margin": 0.03, "debt_to_equity": 4.6})
    assert [i["ratios"] for i in issues] == [["roe", "net_profit_margin", "debt_to_equity"]]


def test_negative_equity_skips_sign_check():
    # Negative equity flips ROE's sign legitimately.
    assert check_consistency({"roe": -0.5, "net_profit_margin": 0.1, "debt_to_equity": -3.0}) == []


def test_pe_with_net_loss():
    issues = check_consistency({"pe_ratio": 40.0, "net_profit_margin": -0.05})
    assert [i["ratios"] for i in issues] == [["pe_ratio", "net_profit_margin"]]


def test_missing_values_skip_checks():
    assert check_consistency({"quick_ratio": 2.0}) == []
