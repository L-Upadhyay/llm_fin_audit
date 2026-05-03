"""
Friendly terminal chatbot for llm_fin_audit.

Pick a stock ticker, then ask any plain-English question about its
financial health. The classical layer (CSP solver, compliance KB,
anomaly detector) is always run up front to give an at-a-glance summary,
and the multi-agent team (DataAgent + AnalysisAgent + ComplianceAgent on
local Ollama llama3.2) handles each free-form question.

Commands inside the chat loop:
    new   — switch to a different ticker
    help  — show example questions
    quit  — exit

Requires:
    ollama serve         (running in another terminal)
    ollama pull llama3.2

Usage:
    python chat.py
"""

import sys

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from src.classical.anomaly_detector import detect_earnings_anomaly
from src.classical.csp_solver import FinancialCSP
from src.classical.knowledge_base import run_compliance_check
from src.data.loader import get_earnings_history, get_financial_ratios
from src.llm.agno_agents import FinancialAnalysisTeam


WELCOME_TEXT = (
    "[bold green]llm_fin_audit[/] — interactive financial-health chatbot\n\n"
    "Pick a stock by ticker (e.g. [bold]AAPL[/], [bold]MSFT[/], [bold]TSLA[/]), "
    "then ask anything in plain English. I'll combine real market data, "
    "classical constraint checks, and a small team of Llama agents to answer.\n\n"
    "Type [bold cyan]help[/] for example questions, "
    "[bold cyan]new[/] to switch tickers, "
    "or [bold cyan]quit[/] to exit."
)

EXAMPLE_QUESTIONS = [
    "Is this company financially healthy?",
    "What are the main risks?",
    "How is the earnings trend?",
    "Should I be concerned about this stock?",
]

VERDICT_STYLE = {"PASS": "green", "WARNING": "yellow", "FAIL": "red"}
SEVERITY_STYLE = {"none": "green", "moderate": "yellow", "severe": "red"}


def fmt(value):
    """Format a numeric ratio for display, or 'n/a' if missing."""
    return "n/a" if value is None else f"{value:.3f}"


_LIVE_BLOCK_HEADER = "**Live market data for"


def strip_live_quote_block(text):
    """
    Remove an auto-prepended live-market-data block from agent text so
    the rich Panel rendering doesn't duplicate the markdown copy.
    """
    if not text:
        return text
    lines = text.split("\n")
    if not lines or not lines[0].lstrip().startswith(_LIVE_BLOCK_HEADER):
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "":
            return "\n".join(lines[i + 1:]).lstrip()
    return ""


def _clean_response(text):
    """
    Strip out raw tool-call JSON that occasionally leaks through Agno's
    team coordinator into the displayed answer.
    """
    if not text:
        return text
    cleaned = []
    for line in text.split("\n"):
        stripped = line.strip()
        if "delegate_task_to_member" in line:
            continue
        if (
            stripped.startswith("{")
            and '"name":' in stripped
            and '"parameters":' in stripped
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------- #
# Display helpers
# ---------------------------------------------------------------------- #

def show_welcome(console):
    console.print(Panel.fit(WELCOME_TEXT, border_style="green"))


def show_help(console):
    lines = ["[bold]Example questions you can ask:[/]\n"]
    for q in EXAMPLE_QUESTIONS:
        lines.append(f"  • [cyan]{q}[/]")
    lines.append("")
    lines.append("[bold]Commands:[/]")
    lines.append("  • [cyan]help[/] — show this list")
    lines.append("  • [cyan]new[/]  — switch to a different ticker")
    lines.append("  • [cyan]quit[/] — exit the chatbot")
    console.print(Panel("\n".join(lines), title="Help", border_style="cyan"))


def show_summary(console, ticker, ratios, earnings):
    """Run the full classical layer and render an 'at a glance' panel."""
    csp_verdict = FinancialCSP().solve(ratios)
    compliance = run_compliance_check(ratios)

    eps_values = list(earnings.get("quarterly_eps", {}).values())
    anomaly = detect_earnings_anomaly(eps_values) if eps_values else None

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("debt_to_equity", fmt(ratios.get("debt_to_equity")))
    table.add_row("current_ratio", fmt(ratios.get("current_ratio")))
    table.add_row("interest_coverage_ratio", fmt(ratios.get("interest_coverage_ratio")))

    csp_color = VERDICT_STYLE.get(csp_verdict, "white")
    kb_color = VERDICT_STYLE.get(compliance["verdict"], "white")
    table.add_row("CSP constraint check", f"[bold {csp_color}]{csp_verdict}[/]")
    table.add_row("Compliance KB", f"[bold {kb_color}]{compliance['verdict']}[/]")

    if anomaly is not None:
        sev_color = SEVERITY_STYLE.get(anomaly["severity"], "white")
        table.add_row("Earnings anomalies", f"[bold {sev_color}]{anomaly['severity']}[/]")
    else:
        table.add_row("Earnings anomalies", "n/a")

    console.print(Panel(
        table,
        title=f"[bold]{ticker} — at a glance[/]",
        border_style="green",
    ))


_REC_BORDER = {"green": "green", "yellow": "yellow", "red": "red"}


def _fmt_price(v):
    return "n/a" if v is None else f"${v:,.2f}"


def _fmt_volume(v):
    return "n/a" if v is None else f"{int(v):,}"


def _fmt_market_cap(v):
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:.2f}T"
    if a >= 1e9:
        return f"${v / 1e9:.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def _fmt_change(c, p):
    """Return (text, rich_color) for the daily price change."""
    if c is None or p is None:
        return "n/a", "white"
    if c > 0:
        return f"▲ +${c:,.2f} (+{p:.2f}%)", "green"
    if c < 0:
        return f"▼ -${abs(c):,.2f} ({p:.2f}%)", "red"
    return "$0.00 (0.00%)", "white"


def show_live_market_data(console, realtime):
    """Render the live yfinance quote as a rich Panel + Table."""
    if not realtime or realtime.get("error"):
        return

    change_text, change_color = _fmt_change(
        realtime.get("price_change"),
        realtime.get("price_change_percent"),
    )

    headline = Text()
    headline.append(_fmt_price(realtime.get("current_price")), style="bold white")
    headline.append("   ")
    headline.append(change_text, style=f"bold {change_color}")

    div = realtime.get("dividend_yield")
    beta = realtime.get("beta")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Previous Close", _fmt_price(realtime.get("previous_close")))
    low, high = realtime.get("day_low"), realtime.get("day_high")
    table.add_row("Today's Range", f"{_fmt_price(low)} – {_fmt_price(high)}")
    yl, yh = realtime.get("fifty_two_week_low"), realtime.get("fifty_two_week_high")
    table.add_row("52-Week Range", f"{_fmt_price(yl)} – {_fmt_price(yh)}")
    table.add_row("Volume", _fmt_volume(realtime.get("volume")))
    table.add_row("Market Cap", _fmt_market_cap(realtime.get("market_cap")))
    table.add_row("Dividend Yield", "n/a" if div is None else f"{div:.2f}%")
    table.add_row("Beta", "n/a" if beta is None else f"{beta:.2f}")
    table.add_row("Next Earnings", realtime.get("next_earnings_date") or "n/a")

    title = (
        f"[bold]Live market data — {realtime.get('ticker', '?')}[/] "
        f"[dim]as of {realtime.get('timestamp', 'now')}[/]"
    )
    console.print(Panel(
        Group(headline, Text(""), table),
        title=title,
        border_style="green",
    ))


def show_recommendation(console, ticker, recommendation):
    """Render the CSP-driven recommendation banner above the answer."""
    if not recommendation:
        return
    label = recommendation.get("label", "")
    emoji = recommendation.get("emoji", "")
    summary = recommendation.get("summary", "")
    color = _REC_BORDER.get(recommendation.get("color"), "white")
    body = (
        f"[bold {color}]{emoji} {label}[/]\n"
        f"[bold]{summary}[/]"
    )
    console.print(Panel(
        body,
        title=f"[bold]Recommendation for {ticker}[/]",
        border_style=color,
    ))


def show_response(console, ticker, result):
    """
    Render a chat response. `result` is the dict returned by
    FinancialAnalysisTeam.run — we render the live-market-data panel
    (if any), the CSP-driven recommendation banner, then the LLM's
    narrative answer.
    """
    if isinstance(result, dict):
        realtime = result.get("realtime")
        if realtime:
            show_live_market_data(console, realtime)
        show_recommendation(console, ticker, result.get("recommendation"))
        text = result.get("text", "")
        if realtime:
            text = strip_live_quote_block(text)
    else:
        text = result

    text = _clean_response(text)
    body = Markdown(text or "_(empty response)_")
    console.print(Panel(
        body,
        title=f"[bold]Answer about {ticker}[/]",
        border_style="magenta",
    ))


def show_error(console, message, hint=None):
    body = f"[red]{message}[/]"
    if hint:
        body += f"\n\n[dim]{hint}[/]"
    console.print(Panel(body, title="[bold red]Error[/]", border_style="red"))


# ---------------------------------------------------------------------- #
# Input loops
# ---------------------------------------------------------------------- #

def prompt_ticker(console):
    """Ask for a ticker until we successfully load data, or the user quits."""
    while True:
        try:
            ticker = Prompt.ask(
                "\n[bold cyan]Enter a stock ticker[/]",
                default="AAPL",
            ).strip().upper()
        except (KeyboardInterrupt, EOFError):
            return None

        if ticker.lower() in ("quit", "exit"):
            return None
        if not ticker:
            continue

        try:
            with console.status(f"[cyan]Loading financial data for {ticker}..."):
                ratios = get_financial_ratios(ticker)
                earnings = get_earnings_history(ticker)
        except Exception as e:  # network errors, bad tickers, yfinance hiccups
            show_error(
                console,
                f"Couldn't load data for {ticker}: {e}",
                hint="Try another ticker (e.g. AAPL, MSFT, TSLA).",
            )
            continue

        return ticker, ratios, earnings


def chat_loop(console, team, ticker):
    """
    Inner loop: take a free-form question, run the agent team, render
    the answer. Returns True if the user wants a new ticker, False to exit.
    """
    while True:
        try:
            user_input = Prompt.ask(f"\n[bold cyan]You ({ticker})[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            return False

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit"):
            console.print("[dim]Goodbye![/]")
            return False
        if cmd == "help":
            show_help(console)
            continue
        if cmd == "new":
            return True  # signal: prompt for a new ticker

        # Otherwise: send to the agent team.
        try:
            with console.status(
                f"[cyan]Thinking about {ticker}... (this may take a moment)"
            ):
                response = team.run(ticker, user_input)
        except Exception as e:
            show_error(
                console,
                f"The agent team hit a problem: {e}",
                hint="Make sure Ollama is running (`ollama serve`) and the "
                     "model is pulled (`ollama pull llama3.2`).",
            )
            continue

        show_response(console, ticker, response)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #

def main():
    console = Console()
    show_welcome(console)

    # One-time team initialization. Slow if Ollama hasn't been touched yet.
    try:
        with console.status("[cyan]Starting up the agent team..."):
            team = FinancialAnalysisTeam()
    except Exception as e:
        show_error(
            console,
            f"Couldn't initialize the agent team: {e}",
            hint="Check that `agno` and `ollama` are installed and that "
                 "Ollama is reachable on localhost.",
        )
        sys.exit(1)

    # Outer loop: choose ticker, show summary, chat. Repeat on 'new'.
    while True:
        loaded = prompt_ticker(console)
        if loaded is None:
            console.print("[dim]Goodbye![/]")
            break

        ticker, ratios, earnings = loaded
        show_summary(console, ticker, ratios, earnings)
        show_help(console)

        keep_going = chat_loop(console, team, ticker)
        if not keep_going:
            break


if __name__ == "__main__":
    main()
