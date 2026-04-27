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
    # D/E above the 2.0 threshold; other ratios fine.
    csp = FinancialCSP()
    ratios = {
        "debt_to_equity": 2.5,
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


if __name__ == "__main__":
    test_healthy_company_passes()
    test_warning_company_warns()
    test_critical_company_fails()
