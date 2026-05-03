"""
Streamlit web app for llm_fin_audit.

Three tabs:
  Analysis  — full classical layer for one ticker (no LLM)
  Compare   — classical layer across AAPL / MSFT / TSLA (no LLM)
  Chat      — Agno multi-agent team backed by Ollama llama3.2

Tabs 1 and 2 deliberately skip the LLM so the UI stays snappy. Only Tab 3
calls the agent team, and only when the user actually sends a message.

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.classical.anomaly_detector import detect_earnings_anomaly
from src.classical.comparator import compare_stocks, rank_stocks
from src.classical.csp_solver import FinancialCSP
from src.classical.knowledge_base import run_compliance_check
from src.data.loader import (
    get_earnings_history,
    get_financial_ratios,
    get_realtime_price,
)


# Verdict / severity styling for table cells. Streamlit's status helpers
# (st.success / st.warning / st.error) handle the colored boxes themselves.
VERDICT_LABEL = {"PASS": "PASS", "WARNING": "WARNING", "FAIL": "FAIL"}
SEVERITY_LABEL = {"none": "none", "moderate": "moderate", "severe": "severe"}


# ---------------------------------------------------------------------- #
# Session state
# ---------------------------------------------------------------------- #

def init_state():
    """One-time defaults so the app survives Streamlit's per-interaction reruns."""
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("team", None)
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("chat_ticker", "AAPL")
    # Compare-tab state
    st.session_state.setdefault("compare_tickers", ["AAPL", "MSFT", "TSLA"])
    st.session_state.setdefault("compare_results", {})  # ticker -> comparator result
    st.session_state.setdefault("earnings_cache", {})   # ticker -> earnings dict


# ---------------------------------------------------------------------- #
# Classical pipeline (no LLM)
# ---------------------------------------------------------------------- #

def run_analysis(ticker):
    """Full classical analysis for one ticker."""
    ratios = get_financial_ratios(ticker)
    csp_verdict = FinancialCSP().solve(ratios)
    kb = run_compliance_check(ratios)

    earnings = get_earnings_history(ticker)
    eps_values = list(earnings.get("quarterly_eps", {}).values())
    anomaly = detect_earnings_anomaly(eps_values) if eps_values else None

    # Live yfinance quote — pulled here so a single Analyze click refreshes
    # the realtime panel along with the rest of the classical layer.
    realtime = get_realtime_price(ticker)

    return {
        "ticker": ticker.upper(),
        "ratios": ratios,
        "csp_verdict": csp_verdict,
        "kb": kb,
        "anomaly": anomaly,
        "earnings": earnings,
        "realtime": realtime,
    }


# ---------------------------------------------------------------------- #
# Display helpers
# ---------------------------------------------------------------------- #

def render_verdict_box(label, verdict):
    """Render a verdict using Streamlit's success/warning/error helpers."""
    text = f"**{label}**: {verdict}"
    if verdict == "PASS":
        st.success(text)
    elif verdict == "WARNING":
        st.warning(text)
    elif verdict == "FAIL":
        st.error(text)
    else:
        st.info(text)


def render_severity_box(severity):
    text = f"**Anomaly severity**: {severity}"
    if severity == "none":
        st.success(text)
    elif severity == "moderate":
        st.warning(text)
    elif severity == "severe":
        st.error(text)
    else:
        st.info(text)


_LIVE_BLOCK_HEADER = "**Live market data for"


def strip_live_quote_block(text):
    """
    Remove an auto-prepended live-market-data block from agent text. The
    chat tab renders the live quote in its own panel, so we drop the
    duplicate markdown copy before showing the LLM's narrative.
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


def _format_price(value):
    """Render a USD price as $1,234.56, or 'n/a' if missing."""
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _format_volume(value):
    """Render share volume with thousands separators."""
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _format_market_cap(value):
    """Render market cap as $X.XXT / $X.XXB / $X.XXM, scaled automatically."""
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


def _format_range(low, high):
    """Format a low–high price range, or 'n/a' if either side is missing."""
    if low is None or high is None:
        return "n/a"
    return f"${low:,.2f} – ${high:,.2f}"


def _format_change(change, change_pct):
    """
    Format the daily change as '▲ +$2.40 (+0.86%)' / '▼ -$1.10 (-0.40%)'.
    Returns ('text', 'color') so the caller can color-code it.
    """
    if change is None or change_pct is None:
        return "n/a", "#666666"
    if change > 0:
        return f"▲ +${change:,.2f} (+{change_pct:.2f}%)", "#2ca02c"
    if change < 0:
        return f"▼ -${abs(change):,.2f} ({change_pct:.2f}%)", "#d62728"
    return f"$0.00 (0.00%)", "#666666"


def _format_dividend_yield(value):
    """Render dividend yield as '0.39%' — yfinance already returns percent units."""
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _format_beta(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def render_live_market_data(realtime):
    """
    Render the live yfinance quote as a clean grid with a green LIVE badge.
    Hidden if `realtime` is empty or yfinance returned an error.
    """
    if not realtime:
        return

    error = realtime.get("error")

    # Header with green LIVE badge.
    st.markdown(
        "<div style='display:flex; align-items:center; gap:10px;'>"
        "<h3 style='margin:0;'>Live Market Data</h3>"
        "<span style='background-color:#2ca02c; color:white; "
        "padding:2px 8px; border-radius:10px; font-size:0.75em; "
        "font-weight:bold;'>LIVE</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    timestamp = realtime.get("timestamp")
    if timestamp:
        st.caption(f"As of {timestamp}")
    else:
        st.caption("Pulled live from yfinance.")

    if error:
        st.warning(f"Couldn't fetch live data: {error}")
        return

    # --- Headline price + delta ---------------------------------------
    price_text = _format_price(realtime.get("current_price"))
    change_text, change_color = _format_change(
        realtime.get("price_change"),
        realtime.get("price_change_percent"),
    )
    st.markdown(
        f"<div style='font-size:2em; font-weight:600;'>{price_text} "
        f"<span style='font-size:0.55em; color:{change_color}; "
        f"font-weight:600;'>{change_text}</span></div>",
        unsafe_allow_html=True,
    )

    # --- Two rows of metrics ------------------------------------------
    row1 = [
        ("Previous Close", _format_price(realtime.get("previous_close"))),
        ("Today's Range", _format_range(realtime.get("day_low"), realtime.get("day_high"))),
        ("52-Week Range", _format_range(realtime.get("fifty_two_week_low"), realtime.get("fifty_two_week_high"))),
        ("Volume", _format_volume(realtime.get("volume"))),
    ]
    row2 = [
        ("Market Cap", _format_market_cap(realtime.get("market_cap"))),
        ("Dividend Yield", _format_dividend_yield(realtime.get("dividend_yield"))),
        ("Beta", _format_beta(realtime.get("beta"))),
        ("Next Earnings", realtime.get("next_earnings_date") or "n/a"),
    ]
    for row in (row1, row2):
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            with col:
                st.metric(label=label, value=value)


def _render_recommendation_banner(recommendation):
    """
    Render the CSP-driven recommendation as a colored Streamlit status box
    above the LLM's answer. Color: green = HOLD, yellow = WATCH, red = AVOID.
    """
    if not recommendation:
        return
    label = recommendation.get("label", "")
    emoji = recommendation.get("emoji", "")
    summary = recommendation.get("summary", "")
    text = f"### {emoji} {label}\n{summary}"
    color = recommendation.get("color")
    if color == "green":
        st.success(text)
    elif color == "yellow":
        st.warning(text)
    elif color == "red":
        st.error(text)
    else:
        st.info(text)


# Display order + labels for the full ratio set. Margin/ROE values come
# back from yfinance as decimals (0.46 = 46%), so they get a percent
# format; the rest are absolute multiples shown to 3 decimals.
_RATIO_DISPLAY = [
    ("debt_to_equity", "Debt-to-Equity", "ratio"),
    ("current_ratio", "Current Ratio", "ratio"),
    ("interest_coverage_ratio", "Interest Coverage", "ratio"),
    ("quick_ratio", "Quick Ratio", "ratio"),
    ("pe_ratio", "P/E Ratio", "ratio"),
    ("roe", "Return on Equity", "percent"),
    ("gross_margin", "Gross Margin", "percent"),
    ("net_profit_margin", "Net Profit Margin", "percent"),
]


def _format_ratio(value, fmt):
    if value is None:
        return "n/a"
    if fmt == "percent":
        return f"{value * 100:.2f}%"
    return f"{value:.3f}"


def render_ratios_table(ratios):
    rows = [
        {"Ratio": label, "Value": _format_ratio(ratios.get(key), fmt)}
        for key, label, fmt in _RATIO_DISPLAY
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def make_ratio_chart(ratios, ticker):
    """
    Two-panel bar chart: absolute multiples on the left, percentage-style
    margins/ROE on the right. The two groups have very different scales,
    so plotting them on a shared y-axis would flatten one of the panels.
    """
    multiples = [
        ("Debt-to-Equity", ratios.get("debt_to_equity")),
        ("Current Ratio", ratios.get("current_ratio")),
        ("Interest Coverage", ratios.get("interest_coverage_ratio")),
        ("Quick Ratio", ratios.get("quick_ratio")),
        ("P/E Ratio", ratios.get("pe_ratio")),
    ]
    margins = [
        ("Return on Equity", ratios.get("roe")),
        ("Gross Margin", ratios.get("gross_margin")),
        ("Net Profit Margin", ratios.get("net_profit_margin")),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def _plot(ax, items, color_cycle, title, ylabel, percent):
        labels = [name for name, _ in items]
        values = [v if v is not None else 0 for _, v in items]
        bars = ax.bar(labels, values, color=color_cycle[: len(items)])
        for bar, (_, raw) in zip(bars, items):
            text = "n/a" if raw is None else (f"{raw * 100:.1f}%" if percent else f"{raw:.2f}")
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                text, ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)

    _plot(
        ax1, multiples,
        ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#bcbd22"],
        f"{ticker} — Multiples & Liquidity", "Value", percent=False,
    )
    _plot(
        ax2, margins,
        ["#ff7f0e", "#d62728", "#8c564b"],
        f"{ticker} — Profitability (%)", "Percent", percent=True,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- #
# Tab renderers
# ---------------------------------------------------------------------- #

def render_analysis_tab():
    st.header("Single-Stock Analysis")

    a = st.session_state.analysis
    if a is None:
        st.info("Enter a ticker in the sidebar and click **Analyze** to start.")
        return

    st.subheader(a["ticker"])

    col1, col2, col3 = st.columns(3)
    with col1:
        render_verdict_box("CSP constraint check", a["csp_verdict"])
    with col2:
        render_verdict_box("Compliance KB", a["kb"]["verdict"])
    with col3:
        if a["anomaly"] is not None:
            render_severity_box(a["anomaly"]["severity"])
        else:
            st.info("**Anomaly severity**: n/a (no earnings data)")

    st.markdown("### Financial Ratios")
    render_ratios_table(a["ratios"])

    # Live market data panel — sits between the ratio table and the chart.
    render_live_market_data(a.get("realtime"))

    st.pyplot(make_ratio_chart(a["ratios"], a["ticker"]))

    st.markdown("### Compliance Rules Triggered")
    rules = a["kb"].get("triggered_rules", [])
    if rules:
        for rule in rules:
            st.write(f"- {rule}")
    else:
        st.write("_None_")

    st.markdown("### Earnings Anomaly Summary")
    if a["anomaly"] is not None:
        st.write(a["anomaly"]["summary"])
        flagged = a["anomaly"].get("anomalies", [])
        if flagged:
            st.markdown("**Flagged quarters:**")
            quarter_dates = list(a["earnings"].get("quarterly_eps", {}).keys())
            for item in flagged:
                idx = item["index"]
                date = quarter_dates[idx] if idx < len(quarter_dates) else f"#{idx}"
                st.write(
                    f"- {date}: EPS = {item['eps']:.2f}, "
                    f"z = {item['z_score']:.2f}"
                )
    else:
        st.write("_No earnings data available._")


# ---------------------------------------------------------------------- #
# Compare-tab metadata and helpers
# ---------------------------------------------------------------------- #

# Numeric ratio metadata: thresholds + plain-English explanation per ratio.
# Thresholds align with the rules in src/classical/csp_solver.py so the bar
# chart colors match the CSP verdict for the same ratio.
RATIO_METADATA = {
    "debt_to_equity": {
        "label": "Debt-to-Equity",
        "explanation": "Measures financial leverage. Lower is generally safer.",
        "ranges_text": "Healthy: below 1.0 • Warning: 1.0–2.0 • Critical: above 2.0",
        "direction": "lower",  # lower numeric value is healthier
        "thresholds": {"healthy_max": 1.0, "warning_max": 2.0},
    },
    "current_ratio": {
        "label": "Current Ratio",
        "explanation": "Measures short-term liquidity. Higher is better.",
        "ranges_text": "Healthy: above 1.5 • Warning: 1.0–1.5 • Critical: below 1.0",
        "direction": "higher",
        "thresholds": {"healthy_min": 1.5, "warning_min": 1.0},
    },
    "interest_coverage_ratio": {
        "label": "Interest Coverage",
        "explanation": "Measures ability to pay interest from operating earnings. Higher is better.",
        "ranges_text": "Healthy: above 3.0 • Warning: 1.5–3.0 • Critical: below 1.5",
        "direction": "higher",
        "thresholds": {"healthy_min": 3.0, "warning_min": 1.5},
    },
    "quick_ratio": {
        "label": "Quick Ratio",
        "explanation": "Stricter liquidity measure that excludes inventory. Higher is better.",
        "ranges_text": "Healthy: above 1.0 • Warning: 0.5–1.0 • Critical: below 0.5",
        "direction": "higher",
        "thresholds": {"healthy_min": 1.0, "warning_min": 0.5},
    },
    "pe_ratio": {
        "label": "P/E Ratio",
        "explanation": "Price relative to trailing earnings. Very high values suggest overvaluation.",
        "ranges_text": "Healthy: below 50 • Warning: 50–100 • Critical: above 100",
        "direction": "lower",
        "thresholds": {"healthy_max": 50.0, "warning_max": 100.0},
    },
    "roe": {
        "label": "Return on Equity",
        "explanation": "Profit generated per dollar of shareholder equity. Higher is better.",
        "ranges_text": "Healthy: above 5% • Warning: 0–5% • Critical: below 0%",
        "direction": "higher",
        "thresholds": {"healthy_min": 0.05, "warning_min": 0.0},
    },
    "gross_margin": {
        "label": "Gross Margin",
        "explanation": "Revenue left after cost of goods sold. Higher means stronger pricing power.",
        "ranges_text": "Healthy: above 20% • Warning: 0–20% • Critical: below 0%",
        "direction": "higher",
        "thresholds": {"healthy_min": 0.20, "warning_min": 0.0},
    },
    "net_profit_margin": {
        "label": "Net Profit Margin",
        "explanation": "Bottom-line profit per dollar of revenue. Higher is better.",
        "ranges_text": "Healthy: above 5% • Warning: 0–5% • Critical: below 0%",
        "direction": "higher",
        "thresholds": {"healthy_min": 0.05, "warning_min": 0.0},
    },
}

# Categorical metadata for verdict-style metrics.
CATEGORICAL_METADATA = {
    "csp_verdict": {
        "label": "CSP Verdict",
        "explanation": "Output of the CSP solver — overall financial-soundness check on the ratios.",
        "value_to_score": {"PASS": 0, "WARNING": 1, "FAIL": 2},
        "score_to_label": {0: "PASS", 1: "WARNING", 2: "FAIL"},
    },
    "anomaly_severity": {
        "label": "Earnings Anomaly Severity",
        "explanation": "Severity of unusual quarters in the last 8 quarters (>2 std deviations from mean).",
        "value_to_score": {"none": 0, "moderate": 1, "severe": 2},
        "score_to_label": {0: "none", 1: "moderate", 2: "severe"},
    },
}

GREEN, YELLOW, RED, GRAY = "#2ca02c", "#ffbf00", "#d62728", "#cccccc"


def _color_for_ratio(metric, value):
    info = RATIO_METADATA[metric]
    if value is None:
        return GRAY
    th = info["thresholds"]
    if info["direction"] == "lower":
        if value <= th["healthy_max"]:
            return GREEN
        if value <= th["warning_max"]:
            return YELLOW
        return RED
    if value >= th["healthy_min"]:
        return GREEN
    if value >= th["warning_min"]:
        return YELLOW
    return RED


def _color_for_categorical(metric, value):
    if value is None:
        return GRAY
    score = CATEGORICAL_METADATA[metric]["value_to_score"].get(value)
    if score is None:
        return GRAY
    return [GREEN, YELLOW, RED][min(score, 2)]


def _ensure_compare_results(tickers):
    """Compute classical results for any tickers not yet cached."""
    cache = st.session_state.compare_results
    missing = [t for t in tickers if t not in cache]
    if not missing:
        return
    with st.spinner(f"Loading {', '.join(missing)}..."):
        try:
            new_results = compare_stocks(missing)
            for r in new_results:
                cache[r["ticker"]] = r
        except Exception as e:
            st.error(f"Couldn't load {', '.join(missing)}: {e}")


def _earnings_for(ticker):
    cache = st.session_state.earnings_cache
    if ticker not in cache:
        try:
            cache[ticker] = get_earnings_history(ticker)
        except Exception:
            cache[ticker] = {"ticker": ticker, "quarterly_eps": {}}
    return cache[ticker]


def _make_ratio_bar_chart(metric, results):
    info = RATIO_METADATA[metric]
    tickers = [r["ticker"] for r in results]
    values = [r["ratios"].get(metric) for r in results]
    plot_values = [v if v is not None else 0 for v in values]
    colors = [_color_for_ratio(metric, v) for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(tickers, plot_values, color=colors)
    for bar, v in zip(bars, values):
        text = f"{v:.2f}" if v is not None else "n/a"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            text, ha="center", va="bottom", fontsize=9,
        )

    th = info["thresholds"]
    if info["direction"] == "lower":
        ax.axhline(th["healthy_max"], color=GREEN, linestyle="--", linewidth=0.7,
                   label=f"healthy ≤ {th['healthy_max']}")
        ax.axhline(th["warning_max"], color=RED, linestyle="--", linewidth=0.7,
                   label=f"critical > {th['warning_max']}")
    else:
        ax.axhline(th["healthy_min"], color=GREEN, linestyle="--", linewidth=0.7,
                   label=f"healthy ≥ {th['healthy_min']}")
        ax.axhline(th["warning_min"], color=RED, linestyle="--", linewidth=0.7,
                   label=f"critical < {th['warning_min']}")

    direction_word = "lower is better" if info["direction"] == "lower" else "higher is better"
    ax.set_title(f"{info['label']} — {direction_word}")
    ax.set_ylabel(info["label"])
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def _make_categorical_bar_chart(metric, results):
    info = CATEGORICAL_METADATA[metric]
    tickers = [r["ticker"] for r in results]
    raw_values = [r.get(metric) for r in results]
    scores = [info["value_to_score"].get(v, 0) for v in raw_values]
    colors = [_color_for_categorical(metric, v) for v in raw_values]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(tickers, scores, color=colors)
    for bar, v in zip(bars, raw_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            v if v is not None else "n/a",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([info["score_to_label"][i] for i in (0, 1, 2)])
    ax.set_ylim(0, 2.4)
    ax.set_title(f"{info['label']} — lower is better")
    fig.tight_layout()
    return fig


def _ratio_info_panel(metric):
    info = RATIO_METADATA[metric]
    direction_word = "lower is better" if info["direction"] == "lower" else "higher is better"
    st.caption(
        f"**{info['label']}** ({direction_word})  \n"
        f"{info['explanation']}  \n"
        f"_{info['ranges_text']}_"
    )


def _categorical_info_panel(metric):
    info = CATEGORICAL_METADATA[metric]
    st.caption(f"**{info['label']}** — {info['explanation']}")


def _earnings_trend_chart(ticker):
    earnings = _earnings_for(ticker)
    eps_dict = earnings.get("quarterly_eps", {})
    if not eps_dict:
        return None

    # quarterly_eps is newest-first; reverse for chronological line.
    dates = list(eps_dict.keys())[::-1]
    values = [eps_dict[d] for d in dates]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(dates, values, marker="o", color="#1f77b4")
    for d, v in zip(dates, values):
        ax.annotate(
            f"{v:.2f}", (d, v), textcoords="offset points",
            xytext=(0, 5), fontsize=8, ha="center",
        )
    ax.set_title(f"{ticker} — Quarterly EPS Trend (oldest → newest)")
    ax.set_ylabel("EPS")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def _csp_explanation(stock_result):
    """Plain-English explanation of why the CSP returned its verdict."""
    verdict = stock_result["csp_verdict"]
    ratios = stock_result["ratios"]
    de = ratios.get("debt_to_equity")
    cr = ratios.get("current_ratio")
    ic = ratios.get("interest_coverage_ratio")

    bullets = []
    if de is not None and de > 2.0:
        bullets.append(
            f"- Debt-to-Equity is **{de:.2f}** (above 2.0) — heavy leverage forces "
            f"this metric to **warning** or **critical**."
        )
    if cr is not None and cr < 1.0:
        bullets.append(
            f"- Current Ratio is **{cr:.2f}** (below 1.0) — short-term obligations "
            f"exceed short-term assets, locking it to **critical**."
        )
    if ic is not None and ic < 1.5:
        bullets.append(
            f"- Interest Coverage is **{ic:.2f}** (below 1.5) — operating earnings "
            f"barely cover interest, locking it to **critical**."
        )

    if not bullets:
        body = "All measured ratios fall in healthy ranges, so the CSP returns **PASS**."
    else:
        body = "\n".join(bullets)

    return f"**CSP verdict: {verdict}**\n\n{body}"


# ---------------------------------------------------------------------- #
# Compare tab
# ---------------------------------------------------------------------- #

def render_compare_tab():
    st.header("Multi-Stock Comparison")
    st.caption(
        "Pick up to 6 tickers, choose the metrics you care about, and "
        "compare side-by-side. Charts and rankings come from the classical "
        "layer (no LLM)."
    )

    # --- Stock list management ----------------------------------------
    st.subheader("Stocks in comparison")
    tickers = st.session_state.compare_tickers

    if tickers:
        # Show each ticker with a Remove button. Cap at 6 columns.
        cols = st.columns(min(len(tickers), 6))
        for i, t in enumerate(list(tickers)):
            with cols[i % len(cols)]:
                st.markdown(f"### {t}")
                if st.button("Remove", key=f"remove_{t}", use_container_width=True):
                    st.session_state.compare_tickers.remove(t)
                    st.rerun()
    else:
        st.info("No stocks selected. Add a ticker below to start.")

    add_col1, add_col2 = st.columns([3, 1])
    with add_col1:
        new_ticker = st.text_input(
            "Add ticker", key="compare_add_input", placeholder="e.g. NVDA",
            label_visibility="collapsed",
        ).strip().upper()
    with add_col2:
        if st.button("Add Stock", use_container_width=True):
            if not new_ticker:
                st.warning("Type a ticker first.")
            elif new_ticker in tickers:
                st.warning(f"{new_ticker} is already in the list.")
            elif len(tickers) >= 6:
                st.warning("Maximum 6 stocks at once. Remove one first.")
            else:
                st.session_state.compare_tickers.append(new_ticker)
                st.rerun()

    if not tickers:
        return

    # --- Lazy load classical results ----------------------------------
    _ensure_compare_results(tickers)
    loaded_results = [
        st.session_state.compare_results[t] for t in tickers
        if t in st.session_state.compare_results
    ]
    if not loaded_results:
        st.warning("No comparison data could be loaded for these tickers.")
        return

    # --- Metric selector ---------------------------------------------
    st.markdown("---")
    st.subheader("Metrics to compare")
    all_metrics = list(RATIO_METADATA.keys()) + list(CATEGORICAL_METADATA.keys())
    metric_labels = {
        **{k: v["label"] for k, v in RATIO_METADATA.items()},
        **{k: v["label"] for k, v in CATEGORICAL_METADATA.items()},
    }
    selected_metrics = st.multiselect(
        "Select metrics", options=all_metrics, default=all_metrics,
        format_func=lambda k: metric_labels[k],
    )

    if not selected_metrics:
        st.info("Pick at least one metric to compare.")
        return

    # --- Per-metric comparison ---------------------------------------
    for metric in selected_metrics:
        st.markdown("---")
        if metric in RATIO_METADATA:
            _ratio_info_panel(metric)
            st.pyplot(_make_ratio_bar_chart(metric, loaded_results))
        else:
            _categorical_info_panel(metric)
            st.pyplot(_make_categorical_bar_chart(metric, loaded_results))

    # --- Ranking summary ---------------------------------------------
    st.markdown("---")
    st.subheader("Overall ranking — healthiest to riskiest")
    ranked = rank_stocks(loaded_results)
    rank_rows = [
        {
            "Rank": r["rank"],
            "Ticker": r["ticker"],
            "Health Score (lower = better)": r["risk_score"],
            "CSP": r["csp_verdict"],
            "KB": r["kb_verdict"],
            "Anomalies": r["anomaly_severity"],
        }
        for r in ranked
    ]
    rank_df = pd.DataFrame(rank_rows)

    def _row_color(score):
        if score <= 1:
            return "background-color: #e8f5e9"  # light green
        if score <= 3:
            return "background-color: #fff8e1"  # light yellow
        return "background-color: #ffebee"      # light red

    styled = rank_df.style.apply(
        lambda row: [_row_color(row["Health Score (lower = better)"])] * len(row),
        axis=1,
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Single-stock deep dive --------------------------------------
    st.markdown("---")
    st.subheader("Deep dive — pick one stock")
    available_tickers = [r["ticker"] for r in loaded_results]
    chosen = st.selectbox("Stock", options=available_tickers, key="deep_dive_select")
    chosen_result = next((r for r in loaded_results if r["ticker"] == chosen), None)
    if chosen_result is None:
        return

    cols = st.columns(3)
    ratios = chosen_result["ratios"]
    for col, (key, info) in zip(cols, RATIO_METADATA.items()):
        with col:
            v = ratios.get(key)
            value_text = f"{v:.3f}" if v is not None else "n/a"
            color = _color_for_ratio(key, v)
            st.markdown(
                f"<div style='padding:12px; border-radius:8px; "
                f"background-color: {color}22; border-left: 5px solid {color};'>"
                f"<b>{info['label']}</b><br>"
                f"<span style='font-size:1.6em'>{value_text}</span><br>"
                f"<span style='font-size:0.85em'>{info['explanation']}</span><br>"
                f"<span style='font-size:0.78em; color:#555;'>{info['ranges_text']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Earnings trend")
    fig = _earnings_trend_chart(chosen)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.write("_No earnings data available._")

    st.markdown("### CSP verdict explained")
    st.markdown(_csp_explanation(chosen_result))


def render_chat_tab():
    st.header("Chat with the Agent Team")
    ticker = st.session_state.chat_ticker
    st.caption(f"Currently chatting about: **{ticker}**  "
               "(set the ticker in the sidebar and click Analyze to switch)")

    # Existing history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if msg.get("realtime"):
                    render_live_market_data(msg["realtime"])
                if msg.get("recommendation"):
                    _render_recommendation_banner(msg["recommendation"])
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about this stock...")
    if not user_input:
        return

    # Show + record user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Lazy-init the team only when the user actually chats.
    if st.session_state.team is None:
        with st.spinner("Starting the agent team (Ollama llama3.2)..."):
            try:
                # Imported lazily so the rest of the app loads even if Agno
                # has issues importing.
                from src.llm.agno_agents import FinancialAnalysisTeam
                st.session_state.team = FinancialAnalysisTeam()
            except Exception as e:
                err = (
                    f"Couldn't start the agent team: {e}\n\n"
                    "Make sure Ollama is running (`ollama serve`) and the "
                    "model is pulled (`ollama pull llama3.2`)."
                )
                with st.chat_message("assistant"):
                    st.error(err)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": err}
                )
                return

    # Run the team
    with st.chat_message("assistant"):
        recommendation = None
        realtime = None
        with st.spinner(f"Thinking about {ticker}... (this may take a minute)"):
            try:
                result = st.session_state.team.run(ticker, user_input)
                response = _clean_response(result.get("text", ""))
                recommendation = result.get("recommendation")
                realtime = result.get("realtime")
                # Avoid double-rendering: the panel below shows the same
                # numbers the agent prepended in markdown form.
                if realtime:
                    response = strip_live_quote_block(response)
            except Exception as e:
                response = (
                    f"The agent team hit a problem: {e}\n\n"
                    "Make sure Ollama is running."
                )

        # Live market data panel (only present for price questions).
        if realtime:
            render_live_market_data(realtime)

        # CSP-driven recommendation banner — rendered above the LLM answer.
        if recommendation:
            _render_recommendation_banner(recommendation)

        st.markdown(response)

    # Persist banner / panel / text so reruns redraw cleanly.
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": response,
            "recommendation": recommendation,
            "realtime": realtime,
        }
    )


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main():
    st.set_page_config(page_title="LLM Financial Audit", layout="wide")
    init_state()

    st.title("LLM Financial Audit System")
    st.caption("Hybrid Classical AI + LLM Analysis")

    # ---------------- Sidebar ----------------
    st.sidebar.header("Controls")
    ticker_input = st.sidebar.text_input("Stock Ticker", value="AAPL").strip().upper()
    ticker = ticker_input or "AAPL"

    if st.sidebar.button("Analyze", use_container_width=True):
        with st.spinner(f"Loading {ticker}..."):
            try:
                st.session_state.analysis = run_analysis(ticker)
                # Reset chat when switching tickers so old answers don't confuse.
                if st.session_state.chat_ticker != ticker:
                    st.session_state.chat_messages = []
                st.session_state.chat_ticker = ticker
            except Exception as e:
                st.sidebar.error(f"Couldn't analyze {ticker}: {e}")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Analysis** and **Compare** use only the classical layer "
        "(CSP + KB + anomaly detector). Manage the comparison list inside "
        "the **Compare** tab. **Chat** uses the multi-agent team and "
        "requires Ollama running locally."
    )

    # ---------------- Tabs ----------------
    tab1, tab2, tab3 = st.tabs(["Analysis", "Compare", "Chat"])
    with tab1:
        render_analysis_tab()
    with tab2:
        render_compare_tab()
    with tab3:
        render_chat_tab()


if __name__ == "__main__":
    main()
