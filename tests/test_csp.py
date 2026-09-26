"""Tests for FinancialCSP across three representative company profiles."""

from src.classical.csp_solver import FinancialCSP


def test_healthy_company_passes():
    csp = FinancialCSP()
    ratios = {
        "debt_to_equity": 0.8,
        "current_ratio": 2.0,
        "interest_coverage_ratio": 10.0,
    }
    verdict = csp.solve(ratios)
    print(f"\nHealthy company verdict: {verdict}")
    assert verdict == "PASS"


def test_warning_company_warns():
    # D/E in the 1.0-2.0 warning band; other ratios fine.
    csp = FinancialCSP()
    ratios = {
        "debt_to_equity": 1.5,
        "current_ratio": 1.5,
        "interest_coverage_ratio": 5.0,
    }
    verdict = csp.solve(ratios)
    print(f"\nWarning company verdict: {verdict}")
    assert verdict == "WARNING"


def test_critical_company_fails():
    # Current ratio below 1.0 -> critical -> FAIL.
    csp = FinancialCSP()
    ratios = {
        "debt_to_equity": 1.5,
        "current_ratio": 0.7,
        "interest_coverage_ratio": 4.0,
    }
    verdict = csp.solve(ratios)
    print(f"\nCritical company verdict: {verdict}")
    assert verdict == "FAIL"


def test_high_leverage_is_critical():
    # D/E above 2.0 is critical on its own — previously it could only
    # ever reach warning because both D/E branches shared one constraint.
    for de in (2.5, 4.6, 50.0):
        ratios = {"debt_to_equity": de, "current_ratio": 2.0}
        assert FinancialCSP().solve(ratios) == "FAIL", de


def test_missing_required_ratios_fail_closed():
    # No data must never read as healthy.
    assert FinancialCSP().solve({}) == "INSUFFICIENT_DATA"
    assert FinancialCSP().solve(
        {"debt_to_equity": None, "current_ratio": None, "roe": 0.3}
    ) == "INSUFFICIENT_DATA"
    assert FinancialCSP().solve({"debt_to_equity": 0.5}) == "INSUFFICIENT_DATA"


def test_optional_ratio_missing_still_solves():
    # interest_coverage is optional (AAPL reports no interest expense).
    ratios = {"debt_to_equity": 0.5, "current_ratio": 2.0,
              "interest_coverage_ratio": None}
    assert FinancialCSP().solve(ratios) == "PASS"


def test_every_ratio_drives_verdict():
    # Each ratio alone, pushed into its critical band, must produce FAIL.
    from src.classical.thresholds import RATIO_THRESHOLDS
    base = {"debt_to_equity": 0.5, "current_ratio": 2.0}
    for metric, th in RATIO_THRESHOLDS.items():
        bad = th["critical"] + 1 if th["direction"] == "lower" else th["critical"] - 1
        assert FinancialCSP().solve({**base, metric: bad}) == "FAIL", metric


if __name__ == "__main__":
    test_healthy_company_passes()
    test_warning_company_warns()
    test_critical_company_fails()
    test_high_leverage_is_critical()
    test_missing_required_ratios_fail_closed()
