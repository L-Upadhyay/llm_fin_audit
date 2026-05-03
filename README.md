# llm_fin_audit

A hybrid classical-AI + LLM system that catches when a language-model financial agent hallucinates numbers, breaks hard constraints, or misreads compliance rules — and corrects it.

> **BU MET CS 664 Term Project — Spring 2026. Student: Lucky Upadhyay.**

## 💡 What it does

LLMs are confidently wrong about finance. They invent ratios, miss covenants, and label distressed companies "healthy." That's fine for chat, fatal for audit.

I built llm_fin_audit to wrap an LLM agent team with a deterministic classical layer that:

- pulls real ratios from yfinance,
- runs a CSP solver to flag constraint violations,
- runs a forward-chaining knowledge base to fire compliance rules,
- runs a statistical anomaly detector on quarterly EPS,

…and only lets the LLM's verdict reach the user after the classical layer signs off. I benchmark three experimental conditions side by side — classical-only, LLM-only, and hybrid — so the cost of removing the safety net is measurable, not hand-waved.

## ✨ Features

- **Financial data loader** (`src/data/loader.py`) — yfinance-backed ratio + EPS extraction with graceful handling of missing line items
- **CSP solver** (`src/classical/csp_solver.py`) — Variable, Constraint, FinancialCSP with AC-3 arc consistency and backtracking + forward checking, written from scratch
- **Forward-chaining knowledge base** (`src/classical/knowledge_base.py`) — Horn-clause Clause / KnowledgeBase engine with a six-rule compliance ruleset
- **Earnings anomaly detector** (`src/classical/anomaly_detector.py`) — z-score outlier flagging, linear-search worst-quarter lookup, half-vs-half trend classification
- **Agno multi-agent team** (`src/llm/agno_agents.py`) — DataAgent / AnalysisAgent / ComplianceAgent on Ollama llama3.2, each calling into the classical layer through @tool wrappers
- **Live Market Data** (`src/data/loader.py`) — real-time price, day change ▲▼, previous close, today's range, 52-week range, volume, market cap, dividend yield, beta, and next earnings date — all pulled live from yfinance with a timestamp
- **HOLD / WATCH / AVOID recommendation system** — verdict driven by the classical CSP result, never the LLM's opinion; rendered as a colored banner in both the web app and the terminal chatbot
- **Multi-stock comparator** (`src/classical/comparator.py`) — side-by-side comparison, composite-risk ranking, and matplotlib charts
- **Evaluation harness** (`src/evaluation/benchmark.py`) — runs classical-only, LLM-only, and hybrid on the same ticker, captures response times, and flags constraint violations
- **CLI surfaces** — `run_demo.py` (rich-terminal classical demo), `run_agent.py` (multi-agent demo), `chat.py` (interactive non-coder chatbot)
- **Streamlit web app** (`app.py`) — three-tab UI: single-stock Analysis, dynamic Compare (add/remove up to 6 tickers, metric multiselect, color-coded thresholds, deep-dive per stock), and an LLM Chat tab gated behind Ollama

All classical components have pytest coverage under `tests/`.

## 🏗️ Architecture

Two cooperating layers — the LLM proposes, the classical layer verifies and corrects.

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
│  ComplianceAgent┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

The LLM cannot output a final verdict until at least one tool call into the classical layer has returned. Every numeric claim in the response is traceable to a yfinance row.

## 🚀 How to run

Clone, install dependencies, then pick an interface.

```bash
git clone https://github.com/L-Upadhyay/llm_fin_audit
cd llm_fin_audit
conda activate spring_2026
pip install -r requirements.txt
```

The web app and chatbot need Ollama running locally with llama3.2 pulled:

```bash
ollama serve
ollama pull llama3.2
```

| Interface | Command | Needs Ollama? |
|---|---|---|
| Web app | `streamlit run app.py` | Only for the Chat tab |
| Terminal chatbot | `python chat.py` | Yes |
| Classical demo | `python run_demo.py` | No |
| Agent demo | `python run_agent.py` | Yes |
| Benchmark | `python -m src.evaluation.benchmark` | Yes |
| Tests | `pytest` | No |

## 🧰 Tech stack

- **Python 3.11**
- **Classical AI** — CSP solver, AC-3, forward checking, forward-chaining KB, statistical anomaly detection, all written against the standard library
- **Data** — yfinance, pandas, numpy
- **Visualization** — matplotlib, rich, streamlit
- **LLM** — agno multi-agent framework, ollama runtime, llama3.2 model
- **Testing** — pytest

## 🎓 Course

BU MET CS 664 — Artificial Intelligence
Prof. Suresh Kalathur · Spring 2026
Student: Lucky Upadhyay (MS Applied Data Analytics)

Modules exercised:
- **Search** — linear-scan worst-quarter lookup, statistical anomaly detection
- **Constraint satisfaction** — AC-3 + backtracking + forward checking on the financial-soundness CSP
- **Knowledge representation & reasoning** — Horn-clause KB with forward chaining
- **Agents & multi-agent systems** — Agno team coordinating specialist agents over a shared classical toolset
- **AIMA hierarchy** — Agent / Environment / Problem patterns mirrored across the classical layer

## 🤖 AI Tool Usage

I built this project with Claude Code (Anthropic) as a coding assistant, in line with BU MET academic-integrity policy on responsible AI use.

I designed and directed every aspect of the project:
- Identified the research problem — LLM hallucination in financial analysis — drawing on my background in financial analysis at EY and PwC
- Designed the hybrid architecture that combines classical AI verification with LLM agents
- Specified the CSP constraint structure, knowledge base rule chains, anomaly thresholds, and multi-agent topology
- Defined the verification contract between the two layers
- Made the design trade-offs and validated each component against the AIMA textbook and course materials
- Reviewed and tested every piece before committing it

Claude Code helped me move faster by:
- Translating my specifications into Python
- Generating test scaffolding for the components I designed
- Drafting documentation that I edited

The architecture, the responsible-AI framing, the choice of techniques, and the evaluation methodology are my own work.
