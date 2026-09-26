"""Regression guard: the verifier on the injected-error suite (offline)."""

from src.evaluation.injection import run


def test_verifier_catches_injected_errors_on_standard_phrasings():
    s = run(docs_per_ticker=2)
    standard = s["recall_by_family"]["standard"]
    assert standard["n"] > 100
    assert standard["recall"] >= 0.95
    assert s["precision"] >= 0.95
    assert s["false_positive_rate_clean"] <= 0.02
