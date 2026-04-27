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
from src.data.loader import get_earnings_history, get_financial_ratios


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

    return {
        "ticker": ticker.upper(),
        "ratios": ratios,
        "csp_verdict": csp_verdict,
        "kb": kb,
        "anomaly": anomaly,
        "earnings": earnings,
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


def render_ratios_table(ratios):
    rows = [
        {"Ratio": "Debt-to-Equity", "Value": ratios.get("debt_to_equity")},
        {"Ratio": "Current Ratio", "Value": ratios.get("current_ratio")},
        {"Ratio": "Interest Coverage", "Value": ratios.get("interest_coverage_ratio")},
    ]
    df = pd.DataFrame(rows)
    df["Value"] = df["Value"].apply(lambda v: f"{v:.3f}" if v is not None else "n/a")
    st.dataframe(df, use_container_width=True, hide_index=True)


def make_ratio_chart(ratios, ticker):
    """Single-ticker bar chart of the three ratios."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Debt-to-Equity", "Current Ratio", "Interest Coverage"]
    values = [
        ratios.get("debt_to_equity") or 0,
        ratios.get("current_ratio") or 0,
        ratios.get("interest_coverage_ratio") or 0,
    ]
    bars = ax.bar(labels, values, color=["#1f77b4", "#2ca02c", "#9467bd"])
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title(f"{ticker} — Key Ratios")
    ax.set_ylabel("Value")
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
        with st.spinner(f"Thinking about {ticker}... (this may take a minute)"):
            try:
                response = st.session_state.team.run(ticker, user_input)
            except Exception as e:
                response = (
                    f"The agent team hit a problem: {e}\n\n"
                    "Make sure Ollama is running."
                )
        st.markdown(response)

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": response}
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
