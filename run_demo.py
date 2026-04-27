"""
Classical-layer demo for llm_fin_audit.

Loads real AAPL data from yfinance, runs all three classical components
(CSP solver, forward-chaining compliance KB, EPS anomaly detector), and
prints each result in a colored rich Panel.

No Agno / Ollama yet — this script exercises only the classical layer
so it can run without a local LLM.

Usage:
    python run_demo.py
    python run_demo.py MSFT      # any ticker
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.classical.anomaly_detector import detect_earnings_anomaly
from src.classical.csp_solver import FinancialCSP
from src.classical.knowledge_base import run_compliance_check
from src.data.loader import get_earnings_history, get_financial_ratios


# Map verdicts/severities to rich color names so panels light up consistently.
VERDICT_STYLE = {
    "PASS": "green",
    "WARNING": "yellow",
    "FAIL": "red",
}
SEVERITY_STYLE = {
    "none": "green",
    "moderate": "yellow",
    "severe": "red",
}


def _fmt(value):
    """Format a numeric ratio for display, or 'n/a' if missing."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def render_ratios(console, ratios):
    """Render the financial-ratios block as a small table inside a panel."""
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Ratio", style="bold")
    table.add_column("Value")

    table.add_row("debt_to_equity", _fmt(ratios.get("debt_to_equity")))
    table.add_row("current_ratio", _fmt(ratios.get("current_ratio")))
    table.add_row("interest_coverage_ratio", _fmt(ratios.get("interest_coverage_ratio")))

    console.print(Panel(
        table,
        title=f"[bold]Financial Ratios — {ratios['ticker']}[/]",
        border_style="cyan",
    ))


def render_csp(console, ratios):
    """Run the CSP and render the verdict in a colored panel."""
    verdict = FinancialCSP().solve(ratios)
    style = VERDICT_STYLE.get(verdict, "white")

    body = Text()
    body.append("Verdict: ", style="bold")
    body.append(verdict, style=f"bold {style}")
    body.append("\n\nThe CSP classifies each ratio as healthy / warning / critical "
                "and applies AC-3 + backtracking with forward checking to find a "
                "consistent assignment. The most-severe status drives the verdict.")

    console.print(Panel(
        body,
        title="[bold]CSP Constraint Check[/]",
        border_style=style,
    ))


def render_compliance(console, ratios):
    """Run the forward-chaining KB and render base facts, derived facts, verdict."""
    result = run_compliance_check(ratios)
    style = VERDICT_STYLE.get(result["verdict"], "white")

    body = Text()
    body.append("Verdict: ", style="bold")
    body.append(result["verdict"], style=f"bold {style}")
    body.append("\n\n")

    body.append("Base facts:\n", style="bold")
    body.append("  " + (", ".join(result["base_facts"]) or "(none)") + "\n\n")

    body.append("Derived facts:\n", style="bold")
    body.append("  " + (", ".join(result["derived_facts"]) or "(none)") + "\n\n")

    body.append("Triggered rules:\n", style="bold")
    if result["triggered_rules"]:
        for rule in result["triggered_rules"]:
            body.append(f"  - {rule}\n")
    else:
        body.append("  (none)\n")

    console.print(Panel(
        body,
        title="[bold]Compliance Knowledge Base[/]",
        border_style=style,
    ))


def render_anomaly(console, earnings):
    """Run the anomaly detector and render the summary in a colored panel."""
    eps_values = list(earnings.get("quarterly_eps", {}).values())
    quarter_labels = list(earnings.get("quarterly_eps", {}).keys())

    if not eps_values:
        console.print(Panel(
            "No earnings history available.",
            title="[bold]Earnings Anomaly Detector[/]",
            border_style="red",
        ))
        return

    result = detect_earnings_anomaly(eps_values)
    style = SEVERITY_STYLE.get(result["severity"], "white")

    body = Text()
    body.append("Severity: ", style="bold")
    body.append(result["severity"], style=f"bold {style}")
    body.append(f"\nMean EPS: {_fmt(result['mean'])}    "
                f"Std: {_fmt(result['std'])}\n\n")

    body.append("Summary:\n", style="bold")
    body.append("  " + result["summary"] + "\n\n")

    body.append("Anomalous quarters:\n", style="bold")
    if result["anomalies"]:
        for a in result["anomalies"]:
            label = quarter_labels[a["index"]] if a["index"] < len(quarter_labels) else f"#{a['index']}"
            body.append(
                f"  - {label}: EPS={a['eps']:.2f}, z={a['z_score']:.2f}\n"
            )
    else:
        body.append("  (none)\n")

    console.print(Panel(
        body,
        title="[bold]Earnings Anomaly Detector[/]",
        border_style=style,
    ))


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    console = Console()

    console.print(Panel.fit(
        f"[bold green]llm_fin_audit[/]  —  Classical-Layer Demo  —  Ticker: [bold]{ticker.upper()}[/]",
        border_style="green",
    ))

    # --- Pull data ----------------------------------------------------
    with console.status("[cyan]Fetching financial data from yfinance..."):
        ratios = get_financial_ratios(ticker)
        earnings = get_earnings_history(ticker)

    # --- Run each classical component, panel by panel -----------------
    render_ratios(console, ratios)
    render_csp(console, ratios)
    render_compliance(console, ratios)
    render_anomaly(console, earnings)


if __name__ == "__main__":
    main()
