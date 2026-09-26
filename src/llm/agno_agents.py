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
from src.formatting import (  # noqa: F401  (several are re-exported for callers)
    RECOMMENDATION_BY_VERDICT,
    format_comparison_block as _format_comparison_block,
    format_live_quote_block as _format_live_quote_block,
    recommendation_for_verdict,
    recommendation_line as _recommendation_line,
    strip_comparison_block,
    strip_live_quote_block,
)
from src.llm.providers import MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_S
from src.llm.routing import (  # noqa: F401
    detect_second_ticker as _detect_second_ticker,
    is_price_question as _is_price_question,
)


def _build_per_ticker_block(ticker: str) -> dict:
    """Fetch ratios + realtime + CSP verdict + recommendation for one ticker."""
    ticker = ticker.upper()
    ratios = get_financial_ratios(ticker)
    realtime = get_realtime_price(ticker)
    csp_verdict = FinancialCSP().solve(ratios)
    recommendation = recommendation_for_verdict(csp_verdict)
    return {
        "ticker": ticker,
        "ratios": ratios,
        "realtime": realtime,
        "csp_verdict": csp_verdict,
        "recommendation": recommendation,
    }


MODEL_ID = "llama3.2"


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
    Detect quarterly EPS anomalies (robust median/MAD z-score above 3.5)
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
        "You are DataAgent. You fetch REAL financial data using your tools "
        "and never invent numbers from memory.",
        "You have exactly two tools: get_financial_ratios_tool and "
        "get_realtime_price_tool. You MUST use one of them.",

        # Hard rule for price-related questions.
        "MANDATORY RULE — REAL-TIME DATA: If the question mentions ANY of "
        "these words or phrases, you MUST call get_realtime_price_tool "
        "with the bound ticker before you reply: "
        "'price', 'current price', 'today', 'open', 'high', 'low', "
        "'volume', '52-week', '52 week', 'fifty-two week', 'market cap', "
        "'market capitalization', 'value', 'worth', 'trading at', "
        "'how much', 'quote', 'stock price', 'share price'. "
        "There is NO exception to this rule. Calling the tool is the only "
        "way to get accurate live numbers — your training data is months "
        "or years old and will be wrong.",

        # Concrete examples so the LLM has a pattern to match.
        "Examples that REQUIRE get_realtime_price_tool:",
        "  - 'What is the current price?' -> call get_realtime_price_tool",
        "  - 'What's NVDA worth today?' -> call get_realtime_price_tool",
        "  - 'How much is the stock trading at?' -> call get_realtime_price_tool",
        "  - 'What is the 52-week high?' -> call get_realtime_price_tool",
        "  - 'What is the market cap?' -> call get_realtime_price_tool",
        "  - 'What was today's open?' -> call get_realtime_price_tool",
        "  - 'What is the trading volume?' -> call get_realtime_price_tool",

        "After the tool returns, quote each number verbatim from the JSON "
        "result. Never round to a memorized number. Never say 'around' or "
        "'approximately' if the JSON has the exact value.",

        "For ratio questions (debt-to-equity, current ratio, P/E, ROE, "
        "margins, etc.) call get_financial_ratios_tool instead.",
        "If a question covers BOTH price and ratios, call BOTH tools.",
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

        # Aggressive routing for live market data — placed at the top so
        # it's the first rule the coordinator considers.
        "MANDATORY ROUTING — LIVE MARKET DATA: If the user's question "
        "mentions ANY of: 'price', 'current price', 'today', 'open', "
        "'high', 'low', 'volume', '52-week', '52 week', 'fifty-two week', "
        "'market cap', 'market capitalization', 'value', 'worth', "
        "'trading at', 'how much', 'quote', 'stock price', or "
        "'share price', you MUST delegate to DataAgent FIRST and require "
        "it to call get_realtime_price_tool. You are FORBIDDEN from "
        "answering price/value/52-week/volume/market-cap questions from "
        "your own memory. Every such number must come from the tool's "
        "JSON output, copied verbatim.",

        "Examples and the tool that must be used:",
        "  - 'What is the current price of NVDA?' -> DataAgent + "
        "get_realtime_price_tool",
        "  - 'How is AAPL valued today?' -> DataAgent + "
        "get_realtime_price_tool",
        "  - 'What is the 52-week high?' -> DataAgent + "
        "get_realtime_price_tool",
        "  - 'What is the market cap?' -> DataAgent + "
        "get_realtime_price_tool",

        "Delegate ratio lookups (D/E, current ratio, P/E, ROE, margins) "
        "to DataAgent (it will call get_financial_ratios_tool).",
        "Delegate constraint and anomaly checks to AnalysisAgent.",
        "Delegate compliance/risk checks to ComplianceAgent.",
        "Synthesize their outputs into a single grounded verdict.",
        "Never invent numbers — every claim must come from a tool result. "
        "If a tool failed, say so explicitly instead of guessing.",
        "ALWAYS end your reply with a markdown section titled "
        "'## Recommendation' on its own line.",
        "The Recommendation section MUST contain exactly one of these "
        "lines, picked from the CSP verdict the user provides at the bottom "
        "of the prompt — do not invent your own:",
        "  - if CSP verdict is PASS:    '✅ HOLD — Ratios are within healthy ranges'",
        "  - if CSP verdict is WARNING: '⚠️ WATCH — Monitor these metrics closely'",
        "  - if CSP verdict is FAIL:    '🔴 AVOID/REVIEW — One or more metrics are critical'",
        "  - if CSP verdict is INSUFFICIENT_DATA: '❔ NO VERDICT — Required ratios "
        "unavailable — cannot assess'. In that case do not describe the company "
        "as healthy or unhealthy.",
        "Copy that line verbatim, including the emoji and dash. Then add one "
        "short sentence explaining the call in your own words.",
    ]

    def __init__(self, model_id: str = MODEL_ID, ticker: str = None):
        # Same guards as the verified pipeline's providers: cap output and
        # widen the repeat window so a looping generation can't hang Ollama.
        self.model = Ollama(
            id=model_id,
            timeout=REQUEST_TIMEOUT_S,
            options={"num_predict": MAX_OUTPUT_TOKENS,
                     "repeat_penalty": 1.1, "repeat_last_n": 256},
        )
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

        If the question mentions a second ticker (e.g. "AAPL vs NVDA"),
        the team enters comparison mode: ratios + realtime + CSP verdict
        are fetched for both tickers, injected into the prompt, and a
        side-by-side markdown table is auto-prepended to the answer.

        Returns:
            {
              "text": str,                # synthesized natural-language reply
              "csp_verdict": str,         # PASS / WARNING / FAIL  (primary)
              "recommendation": dict,     # {label, emoji, summary, color} (primary)
              "ratios": dict,             # ratios for the primary ticker
              "realtime": dict | None,    # primary live quote, single-ticker mode
              "comparison": list | None,  # [primary_block, second_block] in compare mode
            }
        """
        ticker = ticker.upper()
        self._set_ticker(ticker)

        # ---- Comparison mode detection -------------------------------
        second_ticker = _detect_second_ticker(question, ticker)
        comparison = None
        if second_ticker:
            primary_block = _build_per_ticker_block(ticker)
            second_block = _build_per_ticker_block(second_ticker)
            comparison = [primary_block, second_block]
            ratios = primary_block["ratios"]
            csp_verdict = primary_block["csp_verdict"]
            recommendation = primary_block["recommendation"]
        else:
            # Classical layer is the single source of truth for the recommendation.
            ratios = get_financial_ratios(ticker)
            csp_verdict = FinancialCSP().solve(ratios)
            recommendation = recommendation_for_verdict(csp_verdict)

        rec_line = _recommendation_line(recommendation)

        prompt_parts = [
            f"Ticker: {ticker}",
            f"Question: {question}",
            "",
            f"Use the ticker '{ticker}' verbatim in every tool call.",
            "",
            f"CSP verdict (from the classical layer): {csp_verdict}",
            f"Use this verdict to drive the Recommendation section. "
            f"The exact line to copy verbatim is: {rec_line}",
        ]

        # ---- Comparison data injection -------------------------------
        if comparison:
            prompt_parts.extend([
                "",
                f"COMPARISON MODE — two tickers detected: {ticker} and "
                f"{second_ticker}. Address BOTH tickers in your answer. "
                "Provide a side-by-side comparison covering ratios, CSP "
                "verdict, and a recommendation for each.",
                "",
                "Authoritative data for both tickers (quote these numbers "
                "verbatim; never substitute training-data figures):",
                json.dumps(
                    {b["ticker"]: {
                        "ratios": b["ratios"],
                        "realtime": b["realtime"],
                        "csp_verdict": b["csp_verdict"],
                        "recommendation_line": _recommendation_line(b["recommendation"]),
                    } for b in comparison},
                    indent=2,
                    default=str,
                ),
                "",
                "End your answer with a single '## Recommendation' section "
                "containing TWO lines — one per ticker, copied verbatim "
                "from the recommendation_line value above:",
                *[
                    f"  - {b['ticker']}: "
                    f"{_recommendation_line(b['recommendation'])}"
                    for b in comparison
                ],
            ])

        # ---- Live-quote injection (single-ticker price questions) ----
        # Skip in compare mode; the comparison block already includes both
        # tickers' realtime data.
        live_quote = None
        if _is_price_question(question) and not comparison:
            live_quote = get_realtime_price(ticker)
            prompt_parts.extend([
                "",
                "LIVE MARKET DATA (authoritative — quote these exact numbers, "
                "never substitute training-data figures):",
                json.dumps(live_quote, indent=2),
                "",
                "When answering anything about price, today's open/high/low, "
                "volume, 52-week high/low, or market cap, use ONLY the values "
                "above. Format prices as $X.XX and market cap with a "
                "T/B suffix (e.g. $4.82T).",
            ])

        prompt = "\n".join(prompt_parts)
        response = self.team.run(prompt)
        text = response.content or ""

        # ---- Auto-prepend live-quote block (single-ticker mode) ------
        if live_quote is not None:
            quote_block = _format_live_quote_block(live_quote)
            if quote_block:
                text = f"{quote_block}\n\n{text.lstrip()}"

        # ---- Auto-prepend comparison table (compare mode) ------------
        if comparison:
            comp_block = _format_comparison_block(comparison)
            if comp_block:
                text = f"{comp_block}\n\n{text.lstrip()}"

        # ---- Auto-append Recommendation section if missing -----------
        if "## Recommendation" not in text:
            if comparison:
                rec_lines = [
                    f"- **{b['ticker']}**: {_recommendation_line(b['recommendation'])}"
                    for b in comparison
                ]
                text = (
                    f"{text.rstrip()}\n\n## Recommendation\n"
                    + "\n".join(rec_lines)
                )
            else:
                text = f"{text.rstrip()}\n\n## Recommendation\n{rec_line}"

        return {
            "text": text,
            "csp_verdict": csp_verdict,
            "recommendation": recommendation,
            "ratios": ratios,
            "realtime": live_quote,
            "comparison": comparison,
        }
