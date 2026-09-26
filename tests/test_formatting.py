"""Tests for the shared display helpers in src/formatting.py."""

from src.formatting import (
    clean_agent_response,
    fmt_market_cap,
    fmt_price,
    strip_comparison_block,
    strip_live_quote_block,
)


def test_number_formatting():
    assert fmt_price(None) == "n/a"
    assert fmt_price(1234.5) == "$1,234.50"
    assert fmt_market_cap(3.9e12) == "$3.90T"
    assert fmt_market_cap(4.5e9) == "$4.50B"


def test_strip_live_quote_block():
    text = "**Live market data for AAPL — as of now:**\n- Price: $1\n\nAnswer."
    assert strip_live_quote_block(text) == "Answer."
    assert strip_live_quote_block("Answer.") == "Answer."


def test_strip_comparison_block():
    text = "**Comparing AAPL vs MSFT:**\n\n| a | b |\n|---|---|\n\nAnswer."
    assert strip_comparison_block(text) == "Answer."


def test_clean_agent_response_drops_coordinator_noise():
    text = "\n".join([
        '{"name": "delegate_task_to_member", "parameters": {}}',
        "I will delegate this to DataAgent.",
        "DataAgent: fetching ratios",
        "DataAgent's response: done",
        "Apple's current ratio is 0.89.",
    ])
    assert clean_agent_response(text) == "Apple's current ratio is 0.89."
