"""
Agno multi-agent financial analysis team.

Pattern follows Kalathur's step7_agno.py: each specialist is a small
Agent(name, model, tools, instructions, markdown) built by a factory
function, and the coordinator is a Team with members=[...] over the
same model. The tools here wrap our own classical layer (loader, CSP,
KB, anomaly detector) — the LLM never invents numbers, it asks the
classical layer.

Three specialists:
    DataAgent        — fetches raw financial ratios
    AnalysisAgent    — runs CSP constraint check + EPS anomaly detector
    ComplianceAgent  — runs forward-chaining compliance KB

The FinancialAnalysisTeam class wires them together behind a single
run(ticker, question) entry point. Before each run, the actual ticker
is injected into every agent's instruction list so the LLM cannot
fall back to a placeholder string when calling tools.
"""

import json

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.team import Team
from agno.tools import tool

from src.classical.anomaly_detector import detect_earnings_anomaly
from src.classical.csp_solver import FinancialCSP
from src.classical.knowledge_base import run_compliance_check
from src.data.loader import (
    get_earnings_history,
    get_financial_ratios,
    get_realtime_price,
)


MODEL_ID = "llama3.2"


# ---------------------------------------------------------------------------
# Recommendation derived from the CSP verdict
# ---------------------------------------------------------------------------
# The recommendation shown at the top of every chat answer is driven by the
# classical CSP verdict, never by the LLM's opinion. Keeping this as a single
# source of truth means the colored banner in chat.py / app.py and the
# Recommendation section the LLM is instructed to emit always agree.

RECOMMENDATION_BY_VERDICT = {
    "PASS": {
        "label": "HOLD",
        "emoji": "✅",
        "summary": "Ratios are within healthy ranges",
        "color": "green",
    },
    "WARNING": {
        "label": "WATCH",
        "emoji": "⚠️",
        "summary": "Monitor these metrics closely",
        "color": "yellow",
    },
    "FAIL": {
        "label": "AVOID/REVIEW",
        "emoji": "🔴",
        "summary": "One or more metrics are critical",
        "color": "red",
    },
}


def recommendation_for_verdict(verdict: str) -> dict:
    """Return the {label, emoji, summary, color} block for a CSP verdict."""
    return RECOMMENDATION_BY_VERDICT.get(
        verdict,
        {
            "label": "UNKNOWN",
            "emoji": "❔",
            "summary": "Classical layer did not return a verdict",
            "color": "white",
        },
    )


def _recommendation_line(rec: dict) -> str:
    """Render the exact 'Recommendation' line we want at the end of answers."""
    return f"{rec['emoji']} {rec['label']} — {rec['summary']}"


# Sentinels we've seen LLMs hallucinate when an instruction says "the user's
# ticker" without binding a specific value. Tool calls with any of these are
# rejected and the LLM is told to retry with the real symbol.
_PLACEHOLDER_TICKERS = {
    "", "ticker", "<ticker>", "{ticker}", "the ticker", "user's ticker",
    "stock", "symbol", "company", "<symbol>", "your_ticker", "xxx",
}


def _normalize_ticker(ticker: str):
    """
    Return the upper-case ticker if it's a real symbol, or None if the
    LLM passed a placeholder. Tools surface the None case as a JSON error
    so the LLM gets immediate feedback and can retry.
    """
    if ticker is None:
        return None
    cleaned = ticker.strip()
    if cleaned.lower() in _PLACEHOLDER_TICKERS:
        return None
    return cleaned.upper()


def _placeholder_error(ticker: str) -> str:
    return json.dumps({
        "error": (
            f"Invalid ticker '{ticker}'. Pass the actual stock symbol "
            f"(e.g. 'AAPL', 'MSFT'), not a placeholder. Look at the "
            f"'Ticker:' line in the user's message."
        )
    })


# ---------------------------------------------------------------------------
# Custom tools
# Each tool wraps exactly one classical-layer call and returns JSON, so the
# LLM sees a concrete, parseable answer rather than a Python object.
# ---------------------------------------------------------------------------

@tool
def get_financial_ratios_tool(ticker: str) -> str:
    """
    Fetch real financial ratios (debt_to_equity, current_ratio,
    interest_coverage_ratio) for the given stock ticker.

    `ticker` MUST be the literal stock symbol such as 'AAPL', 'MSFT', or
    'TSLA'. Do NOT pass the word 'ticker', '<ticker>', or any other
    placeholder — read the actual symbol from the 'Ticker:' line in the
    user's message and pass that string verbatim.

    Returns a JSON string with the ticker and the three computed ratios.
    """
    symbol = _normalize_ticker(ticker)
    if symbol is None:
        return _placeholder_error(ticker)
    ratios = get_financial_ratios(symbol)
    return json.dumps(ratios)


@tool
def get_realtime_price_tool(ticker: str) -> str:
    """
    Fetch live market data (current price, today's open/high/low, volume,
    52-week high and low, market cap) for the given stock ticker.

    `ticker` MUST be the literal stock symbol such as 'AAPL'. Do NOT pass
    the word 'ticker' or any placeholder — pass the exact symbol from the
    user's message.

    Returns a JSON string with the realtime quote.
    """
    symbol = _normalize_ticker(ticker)
    if symbol is None:
        return _placeholder_error(ticker)
    quote = get_realtime_price(symbol)
    return json.dumps(quote)


@tool
def check_constraints_tool(ticker: str) -> str:
    """
    Run the financial-soundness CSP (FinancialCSP.solve) on the given
    ticker's current ratios and return the verdict.

    `ticker` MUST be the literal stock symbol such as 'AAPL'. Do NOT pass
    the word 'ticker' or any placeholder — pass the exact symbol from
    the user's message.

    Returns a JSON string with the verdict (PASS / WARNING / FAIL) and
    the underlying ratios.
    """
    symbol = _normalize_ticker(ticker)
    if symbol is None:
        return _placeholder_error(ticker)
    ratios = get_financial_ratios(symbol)
    verdict = FinancialCSP().solve(ratios)
    return json.dumps({
        "ticker": symbol,
        "verdict": verdict,
        "ratios": ratios,
    })


@tool
def check_compliance_tool(ticker: str) -> str:
    """
    Run the forward-chaining compliance knowledge base on the given ticker's
    ratios.

    `ticker` MUST be the literal stock symbol such as 'AAPL'. Do NOT pass
    the word 'ticker' or any placeholder — pass the exact symbol from
    the user's message.

    Returns a JSON string listing the rules that fired plus an overall
    verdict.
    """
    symbol = _normalize_ticker(ticker)
    if symbol is None:
        return _placeholder_error(ticker)
    ratios = get_financial_ratios(symbol)
    result = run_compliance_check(ratios)
    return json.dumps({"ticker": symbol, **result})


@tool
def detect_anomalies_tool(ticker: str) -> str:
    """
    Detect quarterly EPS anomalies (>2 standard deviations from the mean)
    for the given ticker.

    `ticker` MUST be the literal stock symbol such as 'AAPL'. Do NOT pass
    the word 'ticker' or any placeholder — pass the exact symbol from
    the user's message.

    Returns a JSON string with flagged quarters, the worst quarter, and
    overall severity.
    """
    symbol = _normalize_ticker(ticker)
    if symbol is None:
        return _placeholder_error(ticker)
    history = get_earnings_history(symbol)
    eps_values = list(history.get("quarterly_eps", {}).values())
    result = detect_earnings_anomaly(eps_values)
    return json.dumps({"ticker": symbol, **result})


# ---------------------------------------------------------------------------
# Specialist agent factories — same shape as step7_agno.py
# Instructions here are the *base* instructions; the team rebinds them with
# a ticker-specific directive before each run.
# ---------------------------------------------------------------------------

def make_data_agent(model, instructions=None):
    """DataAgent — fetches raw financial ratios and live quotes for a ticker."""
    return Agent(
        name="DataAgent",
        model=model,
        tools=[get_financial_ratios_tool, get_realtime_price_tool],
        instructions=list(instructions or []),
        markdown=True,
    )


def make_analysis_agent(model, instructions=None):
    """AnalysisAgent — runs the CSP solver and the EPS anomaly detector."""
    return Agent(
        name="AnalysisAgent",
        model=model,
        tools=[check_constraints_tool, detect_anomalies_tool],
        instructions=list(instructions or []),
        markdown=True,
    )


def make_compliance_agent(model, instructions=None):
    """ComplianceAgent — runs the forward-chaining compliance KB."""
    return Agent(
        name="ComplianceAgent",
        model=model,
        tools=[check_compliance_tool],
        instructions=list(instructions or []),
        markdown=True,
    )


# ---------------------------------------------------------------------------
# Coordinator team
# ---------------------------------------------------------------------------

class FinancialAnalysisTeam:
    """
    Coordinator that delegates to DataAgent, AnalysisAgent, and ComplianceAgent
    and synthesizes their outputs into a single grounded answer.

    Before every run, the team rewrites each agent's instruction list to
    include a strong directive binding the current ticker, so tool calls
    cannot use a placeholder like 'ticker' or '<ticker>'.

    Usage:
        team = FinancialAnalysisTeam()
        answer = team.run("AAPL", "Is this company financially sound?")
        print(answer)
    """

    # Base instructions — never reference a specific ticker. The ticker
    # directive is appended at runtime by _set_ticker.
    BASE_DATA_INSTRUCTIONS = [
        "You fetch real financial data using your tools — never invent numbers.",
        "Call get_financial_ratios_tool with the bound ticker for ratio questions.",
        "Call get_realtime_price_tool with the bound ticker for any question about "
        "current price, today's open/high/low, volume, 52-week high or low, or "
        "market cap. Quote those numbers verbatim — never use figures from your "
        "training data.",
        "Report the raw numbers clearly with no editorializing.",
    ]
    BASE_ANALYSIS_INSTRUCTIONS = [
        "You analyze financial soundness using your classical-AI tools.",
        "Use check_constraints_tool for the CSP verdict on the ratios.",
        "Use detect_anomalies_tool to flag unusual EPS quarters.",
        "Always pass the bound ticker to your tools.",
        "Summarize: ratio verdict, anomaly severity, one-line takeaway.",
    ]
    BASE_COMPLIANCE_INSTRUCTIONS = [
        "You check whether the company triggers any compliance rules.",
        "Call check_compliance_tool exactly once with the bound ticker.",
        "List every triggered rule and report the final verdict.",
    ]
    BASE_TEAM_INSTRUCTIONS = [
        "You lead a financial-analysis team backed by classical AI tools.",
        "Delegate ratio lookups to DataAgent.",
        "Delegate any question about current price, today's open/high/low, "
        "volume, 52-week high/low, or market cap to DataAgent so it can call "
        "get_realtime_price_tool — never answer with prices from memory.",
        "Delegate constraint and anomaly checks to AnalysisAgent.",
        "Delegate compliance/risk checks to ComplianceAgent.",
        "Synthesize their outputs into a single grounded verdict.",
        "Never invent numbers — every claim must come from a tool result.",
        "ALWAYS end your reply with a markdown section titled "
        "'## Recommendation' on its own line.",
        "The Recommendation section MUST contain exactly one of these three "
        "lines, picked from the CSP verdict the user provides at the bottom "
        "of the prompt — do not invent your own:",
        "  - if CSP verdict is PASS:    '✅ HOLD — Ratios are within healthy ranges'",
        "  - if CSP verdict is WARNING: '⚠️ WATCH — Monitor these metrics closely'",
        "  - if CSP verdict is FAIL:    '🔴 AVOID/REVIEW — One or more metrics are critical'",
        "Copy that line verbatim, including the emoji and dash. Then add one "
        "short sentence explaining the call in your own words.",
    ]

    def __init__(self, model_id: str = MODEL_ID, ticker: str = None):
        self.model = Ollama(id=model_id)
        self.data_agent = make_data_agent(self.model)
        self.analysis_agent = make_analysis_agent(self.model)
        self.compliance_agent = make_compliance_agent(self.model)

        self.team = Team(
            name="FinancialAnalysisTeam",
            model=self.model,
            members=[self.data_agent, self.analysis_agent, self.compliance_agent],
            instructions=list(self.BASE_TEAM_INSTRUCTIONS),
            markdown=True,
        )

        self._current_ticker = None
        # Initial pass with no ticker so instructions are populated on the
        # agents (factories couldn't reach the class constants until now).
        self._set_ticker(ticker) if ticker else self._reset_instructions()

    # ------------------------------------------------------------------ #
    # Instruction binding
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ticker_directive(ticker: str) -> str:
        """The instruction line that pins the ticker for every tool call."""
        return (
            f"IMPORTANT: For this conversation the stock ticker is exactly "
            f"'{ticker}'. When calling any tool, pass the literal string "
            f"'{ticker}' as the ticker argument. Do NOT pass the words "
            f"'ticker', 'TICKER', '<ticker>', '{{ticker}}', or any other "
            f"placeholder — pass '{ticker}' verbatim every time."
        )

    def _reset_instructions(self):
        """Restore base instructions on every agent (no ticker bound yet)."""
        self.data_agent.instructions = list(self.BASE_DATA_INSTRUCTIONS)
        self.analysis_agent.instructions = list(self.BASE_ANALYSIS_INSTRUCTIONS)
        self.compliance_agent.instructions = list(self.BASE_COMPLIANCE_INSTRUCTIONS)
        self.team.instructions = list(self.BASE_TEAM_INSTRUCTIONS)
        self._current_ticker = None

    def _set_ticker(self, ticker: str):
        """
        Inject a strong ticker directive into every agent's instructions.

        Always rebuilds from the BASE_* constants so the directive doesn't
        accumulate across runs and the previous ticker doesn't leak in.
        """
        ticker = ticker.upper()
        directive = self._ticker_directive(ticker)
        self.data_agent.instructions = list(self.BASE_DATA_INSTRUCTIONS) + [directive]
        self.analysis_agent.instructions = list(self.BASE_ANALYSIS_INSTRUCTIONS) + [directive]
        self.compliance_agent.instructions = list(self.BASE_COMPLIANCE_INSTRUCTIONS) + [directive]
        self.team.instructions = list(self.BASE_TEAM_INSTRUCTIONS) + [directive]
        self._current_ticker = ticker

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self, ticker: str, question: str) -> dict:
        """
        Bind `ticker` into every agent's instructions, run the classical CSP
        for the official recommendation, and then run the team.

        Returns:
            {
              "text": str,                  # the LLM team's natural-language reply
              "csp_verdict": str,           # PASS / WARNING / FAIL
              "recommendation": dict,       # {label, emoji, summary, color}
              "ratios": dict,               # the ratios CSP ran against
            }
        """
        ticker = ticker.upper()
        self._set_ticker(ticker)

        # Classical layer is the single source of truth for the recommendation.
        ratios = get_financial_ratios(ticker)
        csp_verdict = FinancialCSP().solve(ratios)
        recommendation = recommendation_for_verdict(csp_verdict)
        rec_line = _recommendation_line(recommendation)

        prompt = (
            f"Ticker: {ticker}\n"
            f"Question: {question}\n\n"
            f"Use the ticker '{ticker}' verbatim in every tool call.\n\n"
            f"CSP verdict (from the classical layer): {csp_verdict}\n"
            f"Use this verdict to drive the Recommendation section. "
            f"The exact line to copy verbatim is: {rec_line}"
        )
        response = self.team.run(prompt)
        text = response.content or ""

        # Belt-and-suspenders: if the LLM forgot the section, append it. The
        # banner is rendered separately upstream, so the user never misses it.
        if "## Recommendation" not in text:
            text = f"{text.rstrip()}\n\n## Recommendation\n{rec_line}"

        return {
            "text": text,
            "csp_verdict": csp_verdict,
            "recommendation": recommendation,
            "ratios": ratios,
        }
