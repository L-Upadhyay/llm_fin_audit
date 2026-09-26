"""Tests for the verified pipeline, using a scripted fake model (offline)."""

import json

import pytest

from src.data.facts import facts_from_data
from src.llm.pipeline import AuditPipeline
from src.llm.providers import OllamaProvider, OpenAICompatibleProvider, get_provider


RATIOS = {
    "AAPL": {"debt_to_equity": 1.338, "current_ratio": 0.893, "quick_ratio": 0.85,
             "interest_coverage_ratio": None, "pe_ratio": 33.9, "roe": 1.41,
             "gross_margin": 0.479, "net_profit_margin": 0.272},
    "MSFT": {"debt_to_equity": 0.176, "current_ratio": 1.353, "quick_ratio": 1.14,
             "interest_coverage_ratio": 52.8, "pe_ratio": 24.7, "roe": 0.34,
             "gross_margin": 0.683, "net_profit_margin": 0.393},
}


def snapshot_facts(ticker):
    return facts_from_data(ticker, RATIOS[ticker], {"current_price": 100.0}, {})


class ScriptedProvider:
    """Returns canned answers in order and records the prompts it saw."""
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, system, user, json_schema=None):
        self.prompts.append(user)
        return json.dumps({"answer": self.answers.pop(0)})


def make(provider, **kw):
    return AuditPipeline(provider=provider, fact_source=snapshot_facts, **kw)


def test_correct_first_answer_needs_no_retry():
    p = ScriptedProvider("AAPL's current ratio is 0.89.")
    result = make(p).run("AAPL", "How liquid is it?")
    v = result["verification"]
    assert v["retries"] == 0 and v["final"]["contradicted"] == 0
    assert result["csp_verdict"] == "FAIL"
    assert result["text"].endswith("🔴 AVOID/REVIEW — One or more metrics are critical")
    assert len(p.prompts) == 1


def test_wrong_answer_triggers_retry_with_feedback():
    p = ScriptedProvider("AAPL's current ratio is 1.43.", "AAPL's current ratio is 0.89.")
    result = make(p).run("AAPL", "How liquid is it?")
    v = result["verification"]
    assert v["first_attempt"]["contradicted"] == 1
    assert v["retries"] == 1 and v["final"]["contradicted"] == 0
    assert v["corrections"] == []
    assert "you wrote 1.43, the correct value is 0.89" in p.prompts[1]


def test_still_wrong_after_retry_is_corrected_in_place():
    p = ScriptedProvider("AAPL's current ratio is 1.43.", "AAPL's current ratio is 1.5.")
    result = make(p).run("AAPL", "How liquid is it?")
    assert "0.89 [corrected from 1.5]" in result["answer"]
    assert result["verification"]["corrections"][0]["metric"] == "current_ratio"


def test_unverified_mode_passes_answer_through():
    p = ScriptedProvider("AAPL's current ratio is 1.43.")
    result = make(p, verify=False).run("AAPL", "How liquid is it?")
    assert result["verification"] is None
    assert "1.43" in result["answer"]


def test_comparison_mode_fetches_both_tickers():
    p = ScriptedProvider("AAPL current ratio 0.89; MSFT current ratio 1.35.")
    result = make(p).run("AAPL", "Compare AAPL vs MSFT")
    assert [b["ticker"] for b in result["comparison"]] == ["AAPL", "MSFT"]
    assert result["verification"]["final"]["supported"] == 2
    assert '"ticker": "MSFT"' in p.prompts[0]


def test_non_json_output_falls_back_to_raw_text():
    class RawProvider:
        def complete(self, system, user, json_schema=None):
            return "AAPL's current ratio is 0.89."
    result = make(RawProvider()).run("AAPL", "How liquid is it?")
    assert result["answer"] == "AAPL's current ratio is 0.89."


def test_get_provider_specs():
    assert isinstance(get_provider("ollama:llama3.2"), OllamaProvider)
    assert isinstance(get_provider("openai:gpt-4o-mini"), OpenAICompatibleProvider)
    with pytest.raises(ValueError):
        get_provider("llama3.2")
