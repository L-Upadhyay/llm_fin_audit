"""Tests for the forward-chaining compliance knowledge base."""

from src.classical.knowledge_base import (
    Clause,
    KnowledgeBase,
    run_compliance_check,
)


def test_healthy_company_no_rules_fire():
    # All ratios well within healthy ranges -> no base facts -> no derivations.
    ratios = {
        "debt_to_equity": 0.8,
        "current_ratio": 2.0,
        "interest_coverage_ratio": 10.0,
    }
    result = run_compliance_check(ratios)
    print(f"\nHealthy company: {result}")
    assert result["verdict"] == "PASS"
    assert result["base_facts"] == []
    assert result["derived_facts"] == []
    assert result["triggered_rules"] == []


def test_single_risk_triggers_warning():
    # D/E above threshold but liquidity and solvency are fine.
    # Should derive leverage_risk only — not enough to flag for review.
    ratios = {
        "debt_to_equity": 2.5,
        "current_ratio": 1.5,
        "interest_coverage_ratio": 5.0,
    }
    result = run_compliance_check(ratios)
    print(f"\nSingle-risk company: {result}")
    assert result["verdict"] == "WARNING"
    assert "debt_to_equity_high" in result["base_facts"]
    assert "leverage_risk" in result["derived_facts"]
    assert "high_risk_company" not in result["derived_facts"]
    assert "flag_for_review" not in result["derived_facts"]


def test_multiple_risks_chain_to_review_flag():
    # Leverage + liquidity together -> high_risk_company -> flag_for_review.
    ratios = {
        "debt_to_equity": 2.5,
        "current_ratio": 0.7,
        "interest_coverage_ratio": 10.0,
    }
    result = run_compliance_check(ratios)
    print(f"\nMulti-risk company: {result}")
    assert result["verdict"] == "FAIL"
    assert "leverage_risk" in result["derived_facts"]
    assert "liquidity_risk" in result["derived_facts"]
    assert "high_risk_company" in result["derived_facts"]
    assert "flag_for_review" in result["derived_facts"]


def test_kb_basic_operations():
    # Sanity check the KB primitives independent of the compliance rules.
    kb = KnowledgeBase()
    kb.add_clause(Clause("c", ["a", "b"]))
    kb.tell("a")
    kb.tell("b")
    derived = kb.forward_chain()
    assert derived == ["c"]
    assert kb.ask("c") is True
    assert kb.ask("d") is False


if __name__ == "__main__":
    test_healthy_company_no_rules_fire()
    test_single_risk_triggers_warning()
    test_multiple_risks_chain_to_review_flag()
    test_kb_basic_operations()
