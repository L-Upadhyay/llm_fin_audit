"""Tests for claim extraction and verification (src/verification)."""

from src.verification.verifier import (
    CONTRADICTED,
    SUPPORTED,
    UNVERIFIABLE,
    apply_corrections,
    verify_text,
)


FACTS = {
    "AAPL": {
        "ratios": {
            "debt_to_equity": 1.338, "current_ratio": 0.893, "quick_ratio": 0.906,
            "interest_coverage_ratio": None, "pe_ratio": 33.93, "roe": 1.4147,
            "gross_margin": 0.4786, "net_profit_margin": 0.2715,
        },
        "realtime": {"current_price": 341.07, "market_cap": 3.93e12, "beta": 1.21},
        "csp_verdict": "FAIL",
        "kb_verdict": "WARNING",
    },
    "MSFT": {
        "ratios": {
            "debt_to_equity": 0.176, "current_ratio": 1.353, "quick_ratio": 1.142,
            "interest_coverage_ratio": 52.8, "pe_ratio": 24.68, "roe": 0.3401,
            "gross_margin": 0.6831, "net_profit_margin": 0.3934,
        },
        "realtime": {"current_price": 516.17, "market_cap": 3.84e12},
        "csp_verdict": "WARNING",
    },
    "F": {
        "ratios": {"debt_to_equity": 4.61, "current_ratio": 1.07,
                   "interest_coverage_ratio": -7.87},
        "realtime": {},
        "csp_verdict": "FAIL",
    },
}


def statuses(text, primary="AAPL", facts=None):
    report = verify_text(text, facts or {primary: FACTS[primary]}, primary)
    return [(c.claim.metric, c.status) for c in report.checked]


def test_rounded_values_are_supported():
    assert statuses("Apple's current ratio is 0.89.") == [("current_ratio", SUPPORTED)]
    assert statuses("Apple's current ratio is 0.9.") == [("current_ratio", SUPPORTED)]
    assert statuses("Debt-to-equity stands at 1.34x.") == [("debt_to_equity", SUPPORTED)]


def test_wrong_values_are_contradicted():
    assert statuses("Apple's current ratio is 1.43.") == [("current_ratio", CONTRADICTED)]
    assert statuses("The D/E ratio of 0.12 is low.") == [("debt_to_equity", CONTRADICTED)]


def test_percent_and_fraction_forms():
    assert statuses("Net profit margin is 27.2%.") == [("net_profit_margin", SUPPORTED)]
    assert statuses("Net profit margin is 0.27.") == [("net_profit_margin", SUPPORTED)]
    assert statuses("ROE of 59.14% is impressive.") == [("roe", CONTRADICTED)]
    assert statuses("D/E of 134% is elevated.") == [("debt_to_equity", SUPPORTED)]
    # "gross profit margin" must not be read as net profit margin.
    assert statuses("Gross profit margin is 47.9%.") == [("gross_margin", SUPPORTED)]


def test_comparisons_check_direction():
    assert statuses("The current ratio is below 1.0.") == [("current_ratio", SUPPORTED)]
    assert statuses("The current ratio is above 1.5.") == [("current_ratio", CONTRADICTED)]


def test_benchmarks_are_not_claims():
    # "industry average of 1.5" describes the benchmark, not Apple.
    assert statuses("The current ratio is under the industry average of 1.5.") == []


def test_currency_with_scale_suffix():
    assert statuses("Market cap of $3.93 trillion.") == [("market_cap", SUPPORTED)]
    assert statuses("Market cap of $3.9T.") == [("market_cap", SUPPORTED)]
    assert statuses("Market capitalization is $2.5T.") == [("market_cap", CONTRADICTED)]
    assert statuses("The current price is $341.07.") == [("current_price", SUPPORTED)]


def test_negative_values():
    assert statuses("Interest coverage is -7.9x.", "F") == [("interest_coverage_ratio", SUPPORTED)]
    assert statuses("Interest coverage is 7.9x.", "F") == [("interest_coverage_ratio", CONTRADICTED)]


def test_missing_fact_is_unverifiable():
    # AAPL has no interest coverage in the facts.
    assert statuses("Interest coverage is 30x.") == [("interest_coverage_ratio", UNVERIFIABLE)]


def test_years_and_durations_are_ignored():
    assert statuses("Current ratio (FY2024) of 0.89 over 8 quarters.") == [
        ("current_ratio", SUPPORTED)
    ]


def test_heading_then_bullet_layout():
    text = "**2. Current Ratio**\n\n* Value: 1.43 (as of Q4 2022)\n* Industry: 1.2"
    assert statuses(text) == [("current_ratio", CONTRADICTED)]


def test_table_columns_follow_ticker_header():
    facts = {"AAPL": FACTS["AAPL"], "MSFT": FACTS["MSFT"]}
    text = (
        "| Metric | AAPL | MSFT |\n"
        "|---|---|---|\n"
        "| Current Ratio | 0.89 | 2.31 |\n"
        "| P/E | 33.9 | 24.7 |\n"
    )
    report = verify_text(text, facts, "AAPL")
    got = [(c.claim.ticker, c.claim.metric, c.status) for c in report.checked]
    assert got == [
        ("AAPL", "current_ratio", SUPPORTED),
        ("MSFT", "current_ratio", CONTRADICTED),
        ("AAPL", "pe_ratio", SUPPORTED),
        ("MSFT", "pe_ratio", SUPPORTED),
    ]


def test_ticker_attribution_in_prose():
    facts = {"AAPL": FACTS["AAPL"], "MSFT": FACTS["MSFT"]}
    text = "AAPL has a current ratio of 0.89, while MSFT has a current ratio of 1.35."
    report = verify_text(text, facts, "AAPL")
    assert [(c.claim.ticker, c.status) for c in report.checked] == [
        ("AAPL", SUPPORTED), ("MSFT", SUPPORTED),
    ]


def test_verdict_labels():
    facts = {"AAPL": FACTS["AAPL"], "MSFT": FACTS["MSFT"]}
    report = verify_text("AAPL is rated FAIL. MSFT is rated FAIL too.", facts, "AAPL")
    assert [(c.claim.ticker, c.status) for c in report.checked] == [
        ("AAPL", SUPPORTED), ("MSFT", CONTRADICTED),
    ]


def test_compliance_labels_check_the_kb_verdict():
    report = verify_text(
        "The CSP verdict is FAIL. The compliance verdict is FAIL.",
        {"AAPL": FACTS["AAPL"]}, "AAPL",
    )
    assert [(c.claim.metric, c.status) for c in report.checked] == [
        ("verdict", SUPPORTED), ("compliance_verdict", CONTRADICTED),
    ]


def test_apply_corrections_marks_changes():
    text = "Apple's current ratio is 1.43 and the D/E ratio is 1.34."
    report = verify_text(text, {"AAPL": FACTS["AAPL"]}, "AAPL")
    fixed, corrections = apply_corrections(text, report)
    assert fixed == (
        "Apple's current ratio is 0.89 [corrected from 1.43] "
        "and the D/E ratio is 1.34."
    )
    assert corrections == [{
        "ticker": "AAPL", "metric": "current_ratio", "kind": "value",
        "stated": "1.43", "actual": "0.89",
    }]
    assert "current ratio" in report.feedback().lower()
