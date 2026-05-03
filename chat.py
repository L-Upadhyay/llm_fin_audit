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

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

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
    FinancialAnalysisTeam.run — we render the colored CSP-driven
    recommendation first, then the LLM's narrative answer.
    """
    if isinstance(result, dict):
        show_recommendation(console, ticker, result.get("recommendation"))
        text = result.get("text", "")
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
