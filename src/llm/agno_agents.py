"""
Agno multi-agent financial analysis team.

Pattern follows Kalathur's step7_agno.py: each specialist is a small
Agent(name, model, tools, instructions, markdown) built by a factory
function, and the coordinator is a Team with members=[...] over the
same model. The difference here is that the tools are *our own*
classical-AI components (loader, CSP, KB, anomaly detector) wrapped
with Agno's @tool decorator — the LLM never invents numbers, it
asks the classical layer.

Three specialists:
    DataAgent        — fetches raw financial ratios
    AnalysisAgent    — runs CSP constraint check + EPS anomaly detector
    ComplianceAgent  — runs forward-chaining compliance KB

The FinancialAnalysisTeam class wires the three together behind a
single run(ticker, question) entry point.
"""

import json

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.team import Team
from agno.tools import tool

from src.classical.anomaly_detector import detect_earnings_anomaly
from src.classical.csp_solver import FinancialCSP
from src.classical.knowledge_base import run_compliance_check
from src.data.loader import get_earnings_history, get_financial_ratios


MODEL_ID = "llama3.2"


# ---------------------------------------------------------------------------
# Custom tools
# Each tool wraps exactly one classical-layer call and returns JSON, so the
# LLM sees a concrete, parseable answer rather than a Python object.
# ---------------------------------------------------------------------------

@tool
def get_financial_ratios_tool(ticker: str) -> str:
    """
    Fetch real financial ratios (debt_to_equity, current_ratio,
    interest_coverage_ratio) for `ticker`. Returns a JSON string.
    """
    ratios = get_financial_ratios(ticker)
    return json.dumps(ratios)


@tool
def check_constraints_tool(ticker: str) -> str:
    """
    Run the financial-soundness CSP (FinancialCSP.solve) on the ticker's
    current ratios. Returns JSON with the verdict (PASS / WARNING / FAIL)
    and the underlying ratios that drove it.
    """
    ratios = get_financial_ratios(ticker)
    verdict = FinancialCSP().solve(ratios)
    return json.dumps({
        "ticker": ticker.upper(),
        "verdict": verdict,
        "ratios": ratios,
    })


@tool
def check_compliance_tool(ticker: str) -> str:
    """
    Run the forward-chaining compliance KB on the ticker's ratios.
    Returns JSON listing the rules that fired plus an overall verdict.
    """
    ratios = get_financial_ratios(ticker)
    result = run_compliance_check(ratios)
    return json.dumps({"ticker": ticker.upper(), **result})


@tool
def detect_anomalies_tool(ticker: str) -> str:
    """
    Detect quarterly EPS anomalies (>2 std deviations from the mean) for
    the ticker. Returns JSON with flagged quarters, worst quarter, and
    overall severity.
    """
    history = get_earnings_history(ticker)
    eps_values = list(history.get("quarterly_eps", {}).values())
    result = detect_earnings_anomaly(eps_values)
    return json.dumps({"ticker": ticker.upper(), **result})


# ---------------------------------------------------------------------------
# Specialist agent factories — same shape as step7_agno.py
# ---------------------------------------------------------------------------

def make_data_agent(model):
    """DataAgent — fetches raw financial ratios for a ticker."""
    return Agent(
        name="DataAgent",
        model=model,
        tools=[get_financial_ratios_tool],
        instructions=[
            "You fetch real financial data using your tools — never invent numbers.",
            "Always call get_financial_ratios_tool with the user's ticker.",
            "Report the raw numbers clearly with no editorializing.",
        ],
        markdown=True,
    )


def make_analysis_agent(model):
    """AnalysisAgent — runs the CSP solver and the EPS anomaly detector."""
    return Agent(
        name="AnalysisAgent",
        model=model,
        tools=[check_constraints_tool, detect_anomalies_tool],
        instructions=[
            "You analyze financial soundness using your classical-AI tools.",
            "Use check_constraints_tool for the CSP verdict on the ratios.",
            "Use detect_anomalies_tool to flag unusual EPS quarters.",
            "Summarize: ratio verdict, anomaly severity, one-line takeaway.",
        ],
        markdown=True,
    )


def make_compliance_agent(model):
    """ComplianceAgent — runs the forward-chaining compliance KB."""
    return Agent(
        name="ComplianceAgent",
        model=model,
        tools=[check_compliance_tool],
        instructions=[
            "You check whether the company triggers any compliance rules.",
            "Always call check_compliance_tool with the user's ticker.",
            "List every triggered rule and report the final verdict.",
        ],
        markdown=True,
    )


# ---------------------------------------------------------------------------
# Coordinator team
# ---------------------------------------------------------------------------

class FinancialAnalysisTeam:
    """
    Coordinator that delegates to DataAgent, AnalysisAgent, and ComplianceAgent
    and synthesizes their outputs into a single grounded answer.

    Usage:
        team = FinancialAnalysisTeam()
        answer = team.run("AAPL", "Is this company financially sound?")
        print(answer)
    """

    def __init__(self, model_id: str = MODEL_ID):
        self.model = Ollama(id=model_id)
        self.data_agent = make_data_agent(self.model)
        self.analysis_agent = make_analysis_agent(self.model)
        self.compliance_agent = make_compliance_agent(self.model)

        self.team = Team(
            name="FinancialAnalysisTeam",
            model=self.model,
            members=[self.data_agent, self.analysis_agent, self.compliance_agent],
            instructions=[
                "You lead a financial-analysis team backed by classical AI tools.",
                "Delegate ratio lookups to DataAgent.",
                "Delegate constraint and anomaly checks to AnalysisAgent.",
                "Delegate compliance/risk checks to ComplianceAgent.",
                "Synthesize their outputs into a single grounded verdict.",
                "Never invent numbers — every claim must come from a tool result.",
            ],
            markdown=True,
        )

    def run(self, ticker: str, question: str) -> str:
        """
        Send a ticker + question to the team and return the synthesized
        natural-language response. Requires Ollama to be running locally.
        """
        prompt = f"Ticker: {ticker.upper()}\nQuestion: {question}"
        response = self.team.run(prompt)
        return response.content
