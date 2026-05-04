"""Smoke tests for the financial data loader.

All tests in this module hit the live yfinance API and are marked
`network`. To skip them on a machine without internet access:

    pytest -v -m "not network"
"""

import pytest

from src.data.loader import (
    get_earnings_history,
    get_financial_ratios,
    get_realtime_price,
)


@pytest.mark.network
def test_get_financial_ratios_aapl():
    ratios = get_financial_ratios("AAPL")
    print("\nFinancial ratios for AAPL:")
    for key, value in ratios.items():
        print(f"  {key}: {value}")

    assert ratios["ticker"] == "AAPL"
    assert "debt_to_equity" in ratios
    assert "current_ratio" in ratios
    assert "interest_coverage_ratio" in ratios


@pytest.mark.network
def test_get_earnings_history_aapl():
    earnings = get_earnings_history("AAPL")
    print("\nEarnings history for AAPL:")
    for date, eps in earnings["quarterly_eps"].items():
        print(f"  {date}: {eps}")

    assert earnings["ticker"] == "AAPL"
    assert "quarterly_eps" in earnings


@pytest.mark.network
def test_get_realtime_price_aapl():
    quote = get_realtime_price("AAPL")
    print("\nRealtime quote for AAPL:")
    for key, value in quote.items():
        print(f"  {key}: {value}")

    assert quote["ticker"] == "AAPL"
    assert "current_price" in quote
    assert quote["current_price"] is not None, "current_price should be live"
    assert isinstance(quote["current_price"], (int, float))
    assert quote["current_price"] > 0


if __name__ == "__main__":
    print(get_financial_ratios("AAPL"))
    print(get_earnings_history("AAPL"))
    print(get_realtime_price("AAPL"))
