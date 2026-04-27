"""
Multi-agent demo for llm_fin_audit.

Builds a FinancialAnalysisTeam (DataAgent + AnalysisAgent + ComplianceAgent,
all backed by Ollama llama3.2) and runs two natural-language questions
against AAPL. Each specialist calls into the classical layer through its
@tool functions, so every numeric claim in the response is grounded.

Requires Ollama running locally with the llama3.2 model pulled:
    ollama serve
    ollama pull llama3.2

Usage:
    python run_agent.py
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.llm.agno_agents import FinancialAnalysisTeam


TICKER = "AAPL"
QUESTIONS = [
    "Is Apple financially healthy? Check the ratios, compliance rules, and earnings history.",
    "What are the main financial risks for Apple?",
]


def render_question(console, ticker, question, idx):
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Q{idx}:[/] {question}\n[dim]Ticker:[/] [bold]{ticker}[/]",
        border_style="cyan",
    ))


def render_response(console, response, idx):
    body = Markdown(response or "_(empty response)_")
    console.print(Panel(
        body,
        title=f"[bold]Team Response — Q{idx}[/]",
        border_style="magenta",
    ))


def main():
    console = Console()

    console.print(Panel.fit(
        f"[bold green]llm_fin_audit[/]  —  Multi-Agent Demo  —  Ticker: [bold]{TICKER}[/]",
        border_style="green",
    ))

    with console.status("[cyan]Initializing FinancialAnalysisTeam (Ollama llama3.2)..."):
        team = FinancialAnalysisTeam()

    for i, question in enumerate(QUESTIONS, 1):
        render_question(console, TICKER, question, i)
        with console.status(f"[cyan]Running team on Q{i} (this may take a minute)..."):
            response = team.run(TICKER, question)
        render_response(console, response, i)


if __name__ == "__main__":
    main()
