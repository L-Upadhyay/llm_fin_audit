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


# ---------------------------------------------------------------------- #
# Recommendation labels and auto-prepended blocks
# ---------------------------------------------------------------------- #

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
    "INSUFFICIENT_DATA": {
        "label": "NO VERDICT",
        "emoji": "❔",
        "summary": "Required ratios unavailable — cannot assess",
        "color": "white",
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


def recommendation_line(rec: dict) -> str:
    """Render the exact 'Recommendation' line we want at the end of answers."""
    return f"{rec['emoji']} {rec['label']} — {rec['summary']}"


def format_comparison_block(comparison: list) -> str:
    """
    Render a markdown side-by-side table comparing two tickers.

    Used as the auto-prepended block in compare-mode chat answers, so the
    user always sees a structured comparison even if the LLM rambles.
    """
    if not comparison or len(comparison) < 2:
        return ""

    a, b = comparison[0], comparison[1]
    ta, tb = a["ticker"], b["ticker"]
    ra, rb = a["ratios"], b["ratios"]
    rta, rtb = a["realtime"], b["realtime"]

    price, mc = fmt_price, fmt_market_cap

    def ratio(v):
        return "n/a" if v is None else f"{v:.3f}"

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.2f}%"

    rows = [
        ("Current Price", price(rta.get("current_price")), price(rtb.get("current_price"))),
        ("52-Week Range",
         f"{price(rta.get('fifty_two_week_low'))} – {price(rta.get('fifty_two_week_high'))}",
         f"{price(rtb.get('fifty_two_week_low'))} – {price(rtb.get('fifty_two_week_high'))}"),
        ("Market Cap", mc(rta.get("market_cap")), mc(rtb.get("market_cap"))),
        ("Debt-to-Equity", ratio(ra.get("debt_to_equity")), ratio(rb.get("debt_to_equity"))),
        ("Current Ratio", ratio(ra.get("current_ratio")), ratio(rb.get("current_ratio"))),
        ("P/E", ratio(ra.get("pe_ratio")), ratio(rb.get("pe_ratio"))),
        ("ROE", pct(ra.get("roe")), pct(rb.get("roe"))),
        ("Net Profit Margin", pct(ra.get("net_profit_margin")), pct(rb.get("net_profit_margin"))),
        ("**CSP Verdict**", f"**{a['csp_verdict']}**", f"**{b['csp_verdict']}**"),
        ("**Recommendation**",
         f"{a['recommendation']['emoji']} {a['recommendation']['label']}",
         f"{b['recommendation']['emoji']} {b['recommendation']['label']}"),
    ]

    lines = [
        f"{COMPARISON_HEADER} {ta} vs {tb}:**",
        "",
        f"| Metric | {ta} | {tb} |",
        "|---|---|---|",
    ]
    for label, va, vb in rows:
        lines.append(f"| {label} | {va} | {vb} |")
    return "\n".join(lines)


def format_live_quote_block(quote: dict) -> str:
    """
    Format a get_realtime_price() result as a markdown block.

    Used to prepend authoritative live numbers to chat responses so that
    price questions are answered correctly even if the LLM skips the
    tool call.
    """
    if not quote or quote.get("error"):
        return ""

    price, vol = fmt_price, fmt_volume

    def change(c, p):
        if c is None or p is None:
            return "n/a"
        arrow = "▲" if c > 0 else ("▼" if c < 0 else "•")
        sign = "+" if c > 0 else ""
        return f"{arrow} {sign}${c:,.2f} ({sign}{p:.2f}%)"

    div_yield = quote.get("dividend_yield")
    beta = quote.get("beta")

    lines = [
        f"{LIVE_BLOCK_HEADER} {quote.get('ticker', '?')} — "
        f"as of {quote.get('timestamp', 'now')}:**",
        f"- Current Price: {price(quote.get('current_price'))}  "
        f"{change(quote.get('price_change'), quote.get('price_change_percent'))}",
        f"- Previous Close: {price(quote.get('previous_close'))}",
        f"- Today's Range: {price(quote.get('day_low'))} – {price(quote.get('day_high'))}",
        f"- 52-Week Range: {price(quote.get('fifty_two_week_low'))} – "
        f"{price(quote.get('fifty_two_week_high'))}",
        f"- Volume: {vol(quote.get('volume'))}",
        f"- Market Cap: {fmt_market_cap(quote.get('market_cap'))}",
        f"- Dividend Yield: {'n/a' if div_yield is None else f'{div_yield:.2f}%'}",
        f"- Beta: {'n/a' if beta is None else f'{beta:.2f}'}",
        f"- Next Earnings: {quote.get('next_earnings_date') or 'n/a'}",
    ]
    return "\n".join(lines)
