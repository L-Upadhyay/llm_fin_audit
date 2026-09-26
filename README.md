# llm_fin_audit

[![CI](https://github.com/L-Upadhyay/llm_fin_audit/actions/workflows/ci.yml/badge.svg)](https://github.com/L-Upadhyay/llm_fin_audit/actions/workflows/ci.yml)

A deterministic audit layer between an LLM and anyone relying on its financial analysis. Classical AI (constraint satisfaction, forward-chaining rules, robust anomaly detection) runs on real market data and owns the verdict. The LLM only writes the explanation.

## Why

Asked whether Apple was financially healthy, llama3.2 with no tools answered:

> Debt-to-Equity: **0.12** … Current Ratio: **1.43** … Apple appears financially healthy.

The actual figures at the time, from the annual balance sheet data the system pulls, were **1.34** and **0.89**. A current ratio below 1.0 means short-term liabilities exceed short-term assets. The model was fluent, specific, and wrong. For an equity analyst or an auditor, that is the failure that matters: the model gives confident numbers with nothing to trace them back to.

## How it works

```
                 ┌──────────────── Classical layer (deterministic) ────────────────┐
  yfinance ────► │ loader ──► CSP solver         ──► PASS / WARNING / FAIL /        │
                 │        │                           INSUFFICIENT_DATA             │
                 │        ├─► Horn-clause KB     ──► fired compliance rules        │
                 │        └─► EPS anomaly detector ─► flagged quarters + severity  │
                 └───────────────────────────────┬──────────────────────────────────┘
                                                 │ verdict + facts (JSON)
                                                 ▼
                 ┌──────────────── LLM layer (Agno + Ollama) ──────────────────────┐
                 │ DataAgent · AnalysisAgent · ComplianceAgent → coordinator        │
                 │ writes the narrative; tools call back into the classical layer  │
                 └───────────────────────────────┬──────────────────────────────────┘
                                                 ▼
                     recommendation banner (from the CSP) + narrative
```

- **The verdict never comes from the LLM.** `FinancialCSP` computes it before the model runs. The HOLD / WATCH / AVOID banner is derived from it and appended even if the model omits it.
- **Missing data fails closed.** If leverage or liquidity ratios are unavailable (unknown ticker, rate limit, banks without a current ratio), the answer is `INSUFFICIENT_DATA`, never a default PASS.
- **Live prices are injected, not recalled.** Price questions get a fresh yfinance quote in the prompt and in a structured panel, so the model can't answer from stale training data.

## Components

| Module | What it does |
|---|---|
| `src/classical/thresholds.py` | Single source of truth for ratio cut-offs and verdict labels |
| `src/classical/csp_solver.py` | CSP from scratch: AC-3 arc consistency, backtracking with forward checking |
| `src/classical/knowledge_base.py` | Horn-clause KB with forward chaining; leverage/liquidity/solvency risks chain to `flag_for_review` |
| `src/classical/anomaly_detector.py` | Modified z-score (median/MAD) outlier detection on 8 quarters of EPS |
| `src/classical/comparator.py` | Multi-ticker comparison and composite risk ranking |
| `src/data/loader.py` | yfinance ratios (8), EPS history, live quote |
| `src/llm/agno_agents.py` | Agno team; tools wrap the classical layer; comparison mode for two tickers |
| `src/evaluation/benchmark.py` | Runs classical-only, LLM-only, and hybrid on the same ticker |

## Run it

```bash
git clone https://github.com/L-Upadhyay/llm_fin_audit
cd llm_fin_audit
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
```

The chat features need [Ollama](https://ollama.com) with llama3.2:

```bash
ollama serve
ollama pull llama3.2
```

| Interface | Command | Needs Ollama? |
|---|---|---|
| Web app (Analysis / Compare / Chat tabs) | `streamlit run app.py` | Chat tab only |
| Classical-only terminal demo | `python run_demo.py MSFT` | No |
| Terminal chatbot | `python chat.py` | Yes |
| Agent demo | `python run_agent.py` | Yes |
| Benchmark | `python -m src.evaluation.benchmark` | Yes |
| Offline tests (CI) | `pytest -m "not network"` | No |
| All tests | `pytest` | No (needs internet) |

## Limitations

- **The LLM's narrative is not verified yet.** The verdict is deterministic, but numbers the model writes in its explanation are not checked against the source data. In the May 2026 benchmark, the hybrid condition still produced invented figures for MSFT and a malformed tool-call dump for AAPL. The next milestone is a claim verifier that extracts every number from the response and reconciles it against the facts.
- **The benchmark is small.** It covers 3 tickers, one run each, and uses keyword-based stance detection. Treat `data/benchmark_results.json` as a demo, not an evaluation.
- **Thresholds are generic, not sector-aware.** A single set of cut-offs is applied to every industry. Apple's sub-1.0 current ratio reflects its working-capital model, not distress, but it still triggers a critical flag.
- **Mixed periods.** Debt-to-equity and current ratio come from the latest annual balance sheet, while margins, ROE, and quick ratio come from yfinance's trailing figures.
- **yfinance is a scraper**, not an authoritative filings source.
- **llama3.2 (3B) is unreliable at multi-agent delegation.** This is why prices and verdicts are pre-fetched and injected, not left to tool calls.
- **Not investment advice.** HOLD / WATCH / AVOID are labels for ratio-based risk bands.

## Background

This started as my term project for BU MET CS 664 (Artificial Intelligence, Spring 2026) and I'm continuing to develop it. It draws on five years of financial analysis at PwC and EY. Built with Claude Code as a coding assistant.

## License

MIT
