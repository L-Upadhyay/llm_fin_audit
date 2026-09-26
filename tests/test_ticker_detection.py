"""Tests for comparison-mode ticker detection in the chat layer."""

import pytest

from src.llm.agno_agents import _detect_second_ticker


@pytest.mark.parametrize("question, expected", [
    # Real comparisons
    ("Compare AAPL vs MSFT", "MSFT"),
    ("Is AAPL better than NVDA?", "NVDA"),
    ("Should I buy AAPL or F?", "F"),
    ("aapl or $nvda?", "NVDA"),
    # Ordinary words must not become tickers
    ("What is the debt and cash position?", None),
    ("Is AAPL better than its peers?", None),
    ("Tell me about revenue or margins", None),
    ("Should I buy or sell?", None),
    ("Is the CEO better than the CFO?", None),
    # No comparison signal -> no comparison, even with a second symbol
    ("Tell me about MSFT", None),
])
def test_detect_second_ticker(question, expected):
    assert _detect_second_ticker(question, "AAPL") == expected
