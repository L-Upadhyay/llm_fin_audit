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


def test_half_way_rounding_is_supported():
    facts = {"TGT": {"ratios": {"quick_ratio": 0.255}, "realtime": {}, "csp_verdict": "FAIL"}}
    assert statuses("TGT's quick ratio is 0.26.", "TGT", facts) == [("quick_ratio", SUPPORTED)]


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


def test_comparison_symbols():
    assert statuses("A low current ratio (< 1) signals risk.") == [("current_ratio", SUPPORTED)]
    assert statuses("Current ratio (> 1.5) is comfortable.") == [("current_ratio", CONTRADICTED)]


def test_comparison_threshold_is_exact():
    # "below 1.35" with actual 1.353 is false; the threshold's precision is
    # not rounding slack.
    facts = {"MSFT": FACTS["MSFT"]}
    assert statuses("MSFT's current ratio is below 1.3.", "MSFT", facts) == [
        ("current_ratio", CONTRADICTED)
    ]
    assert statuses("MSFT's current ratio is above 1.", "MSFT", facts) == [
        ("current_ratio", SUPPORTED)
    ]


def test_fraction_without_percent_sign_is_ambiguous():
    # ROE 1.4147 = 141%: "1.41" (fraction) and "141.5" (percent) both hold.
    assert statuses("Return on equity is 1.41.") == [("roe", SUPPORTED)]
    assert statuses("Return on equity is 141.5.") == [("roe", SUPPORTED)]
    assert statuses("Return on equity is 1.41%.") == [("roe", CONTRADICTED)]


def test_heading_attribution_uses_ticker_next_to_the_number():
    facts = {"AAPL": FACTS["AAPL"], "MSFT": FACTS["MSFT"]}
    text = "MSFT has a current ratio of 1.35.\n**Gross Margin**\n* AAPL: 47.9%\n"
    report = verify_text(text, facts, "MSFT")
    assert [(c.claim.ticker, c.status) for c in report.checked] == [
        ("MSFT", SUPPORTED), ("AAPL", SUPPORTED),
    ]


def test_benchmarks_are_not_claims():
    # "industry average of 1.5" describes the benchmark, not Apple.
    assert statuses("The current ratio is under the industry average of 1.5.") == []


def test_currency_with_scale_suffix():
    assert statuses("Market cap of $3.93 trillion.") == [("market_cap", SUPPORTED)]
    assert statuses("Market cap of $3.9T.") == [("market_cap", SUPPORTED)]
    assert statuses("Market capitalization is $2.5T.") == [("market_cap", CONTRADICTED)]
    assert statuses("The current price is $341.07.") == [("current_price", SUPPORTED)]


def test_scale_error_correction_uses_true_magnitude():
    facts = {"AMC": {"ratios": {}, "realtime": {"market_cap": 2.624e9}, "csp_verdict": "FAIL"}}
    text = "AMC's market cap is $2.62T."
    report = verify_text(text, facts, "AMC")
    fixed, _ = apply_corrections(text, report)
    assert fixed == "AMC's market cap is $2.62B [corrected from $2.62T]."


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


def test_nearest_keyword_decides_which_verdict():
    report = verify_text(
        "AAPL has a compliance verdict of WARNING and a CSP verdict of FAIL.",
        {"AAPL": FACTS["AAPL"]}, "AAPL",
    )
    assert [(c.claim.metric, c.status) for c in report.checked] == [
        ("compliance_verdict", SUPPORTED), ("verdict", SUPPORTED),
    ]


def test_per_ratio_labels_check_that_ratios_band():
    # TSLA-style bullets label each ratio; they must be checked against the
    # ratio's own band, not the company's overall FAIL verdict.
    facts = {"TSLA": {"ratios": {"current_ratio": 2.16, "interest_coverage_ratio": 16.6,
                                 "gross_margin": 0.1885},
                      "realtime": {}, "csp_verdict": "FAIL"}}
    text = ("* Current ratio: 2.16 (FAIL)\n"
            "* Interest coverage ratio: 16.62 (PASS)\n"
            "* Gross margin: 18.85% (WARNING)\n")
    report = verify_text(text, facts, "TSLA")
    labels = [(c.claim.metric, c.status) for c in report.checked if c.claim.kind == "status"]
    assert labels == [
        ("current_ratio", CONTRADICTED),
        ("interest_coverage_ratio", SUPPORTED),
        ("gross_margin", SUPPORTED),
    ]


def test_verdict_reference_after_the_label():
    # From the llama3.2 live eval (SBUX / VZ / WMT): all three are correct.
    facts = {"AAPL": FACTS["AAPL"]}  # csp FAIL, compliance WARNING
    for text in (
        "Verdict: FAIL (csp_verdict), WARNING (compliance_verdict).",
        "Compliance_verdict is WARNING, but csp_verdict is FAIL.",
        "Caveat: this reflects the 'WARNING' compliance verdict.",
    ):
        report = verify_text(text, facts, "AAPL")
        assert report.contradicted == [], text


def test_trailing_reference_stays_on_its_line():
    # From the llama3.2 live eval (AMZN / TGT).
    facts = {"AAPL": FACTS["AAPL"]}  # csp FAIL, compliance WARNING
    for text in ("CSP Verdict: FAIL\nCompliance Verdict: WARNING",
                 "Compliance Verdict: WARNING\nCSP Verdict: FAIL"):
        assert verify_text(text, facts, "AAPL").contradicted == [], text


def test_truncation_is_tolerated():
    facts = {"SBUX": {"ratios": {"quick_ratio": 0.496}, "realtime": {}, "csp_verdict": "FAIL"}}
    assert statuses("The quick ratio is 0.49.", "SBUX", facts) == [("quick_ratio", SUPPORTED)]
    assert statuses("The quick ratio is 0.47.", "SBUX", facts) == [("quick_ratio", CONTRADICTED)]


def test_units_must_fit_the_metric():
    # From the llama3.2 live eval (F): "market cap" must not grab a percentage.
    facts = {"F": {"ratios": {}, "realtime": {"market_cap": 5.068e10}, "csp_verdict": "FAIL"}}
    text = "F is valued at $50.68B with a market cap, and a profitability ratio of -18.25%."
    assert verify_text(text, facts, "F").contradicted == []
    # ...and a ratio must not grab a dollar amount.
    assert statuses("The current ratio of $341.07 per share") == []


def test_correction_markers_are_not_reread():
    text = "AAPL's current ratio is 0.89 [corrected from 1.43] and the CSP verdict is FAIL [corrected from PASS]."
    report = verify_text(text, {"AAPL": FACTS["AAPL"]}, "AAPL")
    assert [(c.claim.metric, c.status) for c in report.checked] == [
        ("current_ratio", SUPPORTED), ("verdict", SUPPORTED),
    ]
    assert report.untracked == []


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
