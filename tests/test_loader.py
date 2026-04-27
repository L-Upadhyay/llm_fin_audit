"""Smoke test for the financial data loader."""

from src.data.loader import get_financial_ratios, get_earnings_history


def test_get_financial_ratios_aapl():
    ratios = get_financial_ratios("AAPL")
    print("\nFinancial ratios for AAPL:")
    for key, value in ratios.items():
        print(f"  {key}: {value}")

    assert ratios["ticker"] == "AAPL"
    assert "debt_to_equity" in ratios
    assert "current_ratio" in ratios
    assert "interest_coverage_ratio" in ratios


def test_get_earnings_history_aapl():
    earnings = get_earnings_history("AAPL")
    print("\nEarnings history for AAPL:")
    for date, eps in earnings["quarterly_eps"].items():
        print(f"  {date}: {eps}")

    assert earnings["ticker"] == "AAPL"
    assert "quarterly_eps" in earnings


if __name__ == "__main__":
    print(get_financial_ratios("AAPL"))
    print(get_earnings_history("AAPL"))
