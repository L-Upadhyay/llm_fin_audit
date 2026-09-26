"""
Shared display helpers for the agent layer, the terminal chatbot, and the
Streamlit app.

Kept free of agno/streamlit/rich imports so any surface can use it without
pulling in the others.
"""

import re


# ---------------------------------------------------------------------- #
# Number formatting
# ---------------------------------------------------------------------- #

def fmt_price(value):
    """Render a USD price as $1,234.56, or 'n/a' if missing."""
    return "n/a" if value is None else f"${value:,.2f}"


def fmt_volume(value):
    """Render share volume with thousands separators, or 'n/a'."""
    return "n/a" if value is None else f"{int(value):,}"


def fmt_market_cap(value):
    """Render market cap as $X.XXT / $X.XXB / $X.XXM, or 'n/a'."""
    if value is None:
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


# ---------------------------------------------------------------------- #
# Auto-prepended blocks in agent responses
# ---------------------------------------------------------------------- #

# FinancialAnalysisTeam.run prepends these blocks so the authoritative
# numbers always appear, even if the LLM skips a tool call. UIs that render
# the same data in a structured panel strip them to avoid showing it twice.
LIVE_BLOCK_HEADER = "**Live market data for"
COMPARISON_HEADER = "**Comparing"


def strip_live_quote_block(text):
    """Remove the auto-prepended live-market-data block, if present."""
    if not text:
        return text
    lines = text.split("\n")
    if not lines or not lines[0].lstrip().startswith(LIVE_BLOCK_HEADER):
        return text
    # The block ends at the first blank line after the header.
    for i in range(1, len(lines)):
        if lines[i].strip() == "":
            return "\n".join(lines[i + 1:]).lstrip()
    return ""


def strip_comparison_block(text):
    """Remove the auto-prepended two-ticker comparison table, if present."""
    if not text:
        return text
    lines = text.split("\n")
    if not lines or not lines[0].lstrip().startswith(COMPARISON_HEADER):
        return text
    # The block ends at the first blank line AFTER the table (the blank line
    # between the header and the table is part of the block).
    seen_table = False
    for i in range(1, len(lines)):
        if lines[i].lstrip().startswith("|"):
            seen_table = True
        elif seen_table and lines[i].strip() == "":
            return "\n".join(lines[i + 1:]).lstrip()
    return ""


# ---------------------------------------------------------------------- #
# Coordinator noise
# ---------------------------------------------------------------------- #

_AGENT_PREFIXES = (
    "dataagent:", "analysisagent:", "complianceagent:",
    "data agent:", "analysis agent:", "compliance agent:",
    "data_agent:", "analysis_agent:", "compliance_agent:",
)

# Matches "DataAgent's response:", "AAPL's response:", "NVDA's response:",
# etc. — internal coordination labels we never want to surface.
_RESPONSE_LABEL_RE = re.compile(r"^[A-Za-z][\w\-]{0,9}'s\s+response:", re.IGNORECASE)


def clean_agent_response(text):
    """
    Strip raw tool-call JSON and internal coordination noise that leaks
    through Agno's team coordinator before displaying the answer.

    Removed:
      - tool-call JSON like {"name": "...", "parameters": {...}}
      - lines containing 'delegate' (any form, case-insensitive)
      - lines starting with an agent label like 'DataAgent:'
      - lines starting with "<X>'s response:" labels (DataAgent's, AAPL's, ...)
    """
    if not text:
        return text
    cleaned = []
    for line in text.split("\n"):
        stripped = line.strip()
        low = stripped.lower()
        if (
            stripped.startswith("{")
            and '"name":' in stripped
            and '"parameters":' in stripped
        ):
            continue
        # Covers delegate_task_to_member and natural-language delegation chatter.
        if "delegate" in low:
            continue
        if any(low.startswith(p) for p in _AGENT_PREFIXES):
            continue
        if _RESPONSE_LABEL_RE.match(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()
