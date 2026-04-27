# 🧮 llm_fin_audit

A hybrid classical-AI + LLM system that catches when a language-model financial agent hallucinates numbers, breaks hard constraints, or misreads compliance rules — and corrects it.

---

## 💡 What it does

LLMs are confidently wrong about finance. They invent ratios, miss covenants, and label distressed companies "healthy." That's fine for chat, fatal for audit.

`llm_fin_audit` wraps an LLM agent team with a deterministic classical layer that:

- pulls **real** ratios from yfinance,
- runs a **CSP solver** to flag constraint violations,
- runs a **forward-chaining knowledge base** to fire compliance rules,
- runs a **statistical anomaly detector** on quarterly EPS,

…and only lets the LLM's verdict reach the user after the classical layer signs off. Three experimental conditions — *classical-only*, *LLM-only*, *hybrid* — are benchmarked side by side so the cost of removing the safety net is measurable, not hand-waved.

---

## ✨ Features

1. **Financial data loader** (`src/data/loader.py`) — yfinance-backed ratio + EPS extraction with graceful handling of missing line items.
2. **CSP solver** (`src/classical/csp_solver.py`) — `Variable`, `Constraint`, `FinancialCSP` with AC-3 arc consistency and backtracking + forward checking, written from scratch.
3. **Forward-chaining knowledge base** (`src/classical/knowledge_base.py`) — Horn-clause `Clause` / `KnowledgeBase` engine with a six-rule compliance ruleset.
4. **Earnings anomaly detector** (`src/classical/anomaly_detector.py`) — z-score outlier flagging, linear-search worst-quarter lookup, half-vs-half trend classification.
5. **Agno multi-agent team** (`src/llm/agno_agents.py`) — `DataAgent` / `AnalysisAgent` / `ComplianceAgent` on Ollama llama3.2, each calling into the classical layer through `@tool` wrappers, with hardened ticker injection so the LLM cannot pass placeholder strings.
6. **Multi-stock comparator** (`src/classical/comparator.py`) — side-by-side comparison, composite-risk ranking, and matplotlib charts.
7. **Evaluation harness** (`src/evaluation/benchmark.py`) — runs *classical-only*, *LLM-only*, and *hybrid* on the same ticker, captures response times, and flags constraint violations heuristically.
8. **CLI surfaces** — `run_demo.py` (rich-terminal classical demo), `run_agent.py` (multi-agent demo), `chat.py` (interactive non-coder chatbot).
9. **Streamlit web app** (`app.py`) — three-tab UI: single-stock **Analysis**, dynamic **Compare** (add/remove up to 6 tickers, metric multiselect, color-coded thresholds, deep-dive per stock), and an LLM **Chat** tab gated behind Ollama.

All classical components and tests live under `src/` and `tests/` with `pytest` coverage.

---

## 🏗️ Architecture

Two cooperating layers — the LLM proposes, the classical layer verifies and corrects.

```
┌─────────────────────── Classical layer (deterministic) ────────────────────────┐
│  loader (yfinance) ─► CSP solver       ─► PASS / WARNING / FAIL                │
│                    ─► Knowledge base   ─► triggered compliance rules           │
│                    ─► Anomaly detector ─► severity + flagged quarters          │
└────────────────────────────────────────────────────────────────────────────────┘
                                     ▲
                                     │  @tool wrappers (return JSON)
                                     │
┌──────────────────────── LLM layer (Agno + Ollama) ──────────────────────────┐
│  DataAgent  ──┐                                                             │
│  AnalysisAgent├─► FinancialAnalysisTeam coordinator ─► grounded answer      │
│  ComplianceAgent┘  (ticker injected per run; no placeholder hallucinations) │
└─────────────────────────────────────────────────────────────────────────────┘
```

The LLM cannot output a final verdict until at least one tool call into the classical layer has returned. Every numeric claim in the response is traceable to a yfinance row.

---

## 🚀 How to run

Clone, install dependencies, then pick your interface.

```bash
conda activate spring_2026
pip install -r requirements.txt
```

The web app and chatbot need Ollama with the `llama3.2` model:

```bash
ollama serve
ollama pull llama3.2
```

| | Command | Needs Ollama? |
|---|---|---|
| Web app | `streamlit run app.py` | Only for the **Chat** tab |
| Terminal chatbot | `python chat.py` | Yes |
| Classical demo | `python run_demo.py [TICKER]` | No |
| Agent demo | `python run_agent.py` | Yes |
| Benchmark | `python -m src.evaluation.benchmark` | Yes |
| Tests | `pytest` | No (LLM tests skipped) |

---

## 🧰 Tech stack

- **Python 3.11**
- **Classical AI** — CSP solver, AC-3, forward checking, forward-chaining KB, statistical anomaly detection, all written against the standard library
- **Data** — `yfinance`, `pandas`, `numpy`
- **Visualization** — `matplotlib`, `rich`, `streamlit`
- **LLM** — `agno` multi-agent framework, `ollama` runtime, `llama3.2` model
- **Testing** — `pytest`

---

## 🎓 Course

**BU MET CS 664 — Artificial Intelligence**
Prof. Suresh Kalathur · Spring 2026
Student: **Lucky Upadhyay** (MS Applied Data Analytics)

Modules exercised:

- **Search** — linear-scan worst-quarter lookup, statistical anomaly detection
- **Constraint satisfaction** — AC-3 + backtracking + forward checking on the financial-soundness CSP
- **Knowledge representation & reasoning** — Horn-clause KB with forward chaining
- **Agents & multi-agent systems** — Agno team coordinating specialist agents over a shared classical toolset
- **AIMA hierarchy** — Agent / Environment / Problem patterns mirrored across the classical layer

---

## 🤖 AI Tool Disclosure

This project was scaffolded with the help of **Claude Code** (Anthropic). AI assistance was used for:

- File scaffolding and boilerplate (project layout, test stubs, Streamlit wiring),
- Test generation and refactoring suggestions,
- Documentation drafting (this README, in-code comments).

**All algorithmic design is the student's own work** — the CSP search strategy, AC-3 implementation, forward-chaining inference, anomaly thresholds, multi-agent topology, and the hybrid LLM/classical architecture were specified, debugged, and validated by the student against the AIMA / Kalathur source material. Use of AI tooling is disclosed in line with BU MET academic-integrity policy on responsible AI use.
