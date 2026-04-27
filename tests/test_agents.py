"""
Smoke test for the Agno multi-agent team.

We only verify construction here — running a real query needs Ollama to be
live, which is out of scope for an automated test suite.
"""

from src.llm.agno_agents import (
    FinancialAnalysisTeam,
    check_compliance_tool,
    check_constraints_tool,
    detect_anomalies_tool,
    get_financial_ratios_tool,
)


def test_team_initializes():
    team = FinancialAnalysisTeam()

    # All three specialists got built.
    assert team.data_agent is not None
    assert team.analysis_agent is not None
    assert team.compliance_agent is not None
    assert team.team is not None

    # Each specialist has its tools registered.
    assert team.data_agent.tools, "DataAgent has no tools"
    assert team.analysis_agent.tools, "AnalysisAgent has no tools"
    assert team.compliance_agent.tools, "ComplianceAgent has no tools"

    # Names match the project's agent roster.
    assert team.data_agent.name == "DataAgent"
    assert team.analysis_agent.name == "AnalysisAgent"
    assert team.compliance_agent.name == "ComplianceAgent"

    print("\nFinancialAnalysisTeam initialized OK.")


def test_tools_exist():
    # The four custom tools should at least be importable as objects.
    assert get_financial_ratios_tool is not None
    assert check_constraints_tool is not None
    assert check_compliance_tool is not None
    assert detect_anomalies_tool is not None


if __name__ == "__main__":
    test_tools_exist()
    test_team_initializes()
