# llm_fin_audit — CLAUDE.md

## Project Identity
- Name: llm_fin_audit
- GitHub: https://github.com/L-Upadhyay/llm_fin_audit
- Course: BU MET CS 664 Artificial Intelligence, Prof. Suresh Kalathur, Spring 2026
- Student: Lucky Upadhyay
- Conda env: spring_2026
- Python: 3.11

## What This Project Does
Hybrid classical-AI + LLM system that catches when LLM financial agents hallucinate numbers, break hard constraints, or misread compliance rules — and corrects them using a deterministic classical layer.

Two cooperating layers:
1. Classical layer (deterministic) — CSP solver, KB forward chaining, anomaly detector, comparator
2. LLM layer (Agno + Ollama llama3.2) — multi-agent team that CANNOT output a verdict until classical layer signs off

## File Structure
src/
data/loader.py          — yfinance ratio extraction (8 ratios) + get_realtime_price() (16 fields)
classical/
csp_solver.py         — Variable, Constraint, FinancialCSP with AC-3 + backtracking + forward checking
knowledge_base.py     — Horn-clause Clause/KnowledgeBase with forward chaining, 6 compliance rules
anomaly_detector.py   — z-score outlier flagging, worst-quarter lookup, trend classification
comparator.py         — multi-stock comparison, composite risk ranking, matplotlib charts
llm/
agno_agents.py        — DataAgent, AnalysisAgent, ComplianceAgent, FinancialAnalysisTeam
evaluation/
benchmark.py          — 3-condition evaluation harness (classical-only, LLM-only, hybrid)
tests/                    — 22 pytest tests, all passing
app.py                    — Streamlit web app (Analysis, Compare, Chat tabs)
chat.py                   — Terminal interactive chatbot
run_demo.py               — Terminal classical-only demo
run_agent.py              — Terminal agent demo

## Critical Rules — DO NOT VIOLATE

### Classical layer
- NO external solver libraries — no python-constraint, no ortools, no networkx
- CSP written from scratch using only standard library
- KB written from scratch using only standard library
- All classical components must remain independently testable

### LLM layer
- LLM cannot output financial verdict without CSP solver verifying first
- HOLD/WATCH/AVOID recommendation ALWAYS driven by CSP verdict, never LLM opinion
- HOLD = CSP PASS (green), WATCH = CSP WARNING (yellow), AVOID = CSP FAIL (red)
- Real-time price ALWAYS pre-fetched from yfinance and injected — never let LLM answer price questions from training data
- Strip all raw JSON delegation text (delegate_task_to_member, DataAgent:, AnalysisAgent:, etc.) from responses before displaying

### Tests
- Always run pytest after any change
- Must maintain 22 tests passing
- Never break existing tests to add new features
- 17 tests run fully offline (CSP, KB, anomaly, comparator, agent construction, stance/violation helpers)
- 5 tests need yfinance network access — marked @pytest.mark.network
- Offline run: pytest -v -m "not network" → 17 tests
- Full run:    pytest -v → 22 tests, requires internet

### Git
- Commit after every working feature
- Push to origin main
- Clear descriptive commit messages

## Financial Ratios (8 total)
1. debt_to_equity — from ticker.info["debtToEquity"] / 100
2. current_ratio — from ticker.info["currentRatio"]
3. interest_coverage_ratio — calculated from financials
4. quick_ratio — from ticker.info["quickRatio"]
5. pe_ratio — from ticker.info["trailingPE"]
6. roe — from ticker.info["returnOnEquity"]
7. gross_margin — from ticker.info["grossMargins"]
8. net_profit_margin — from ticker.info["profitMargins"]

## Live Market Data Fields (16 total)
current_price, open, day_high, day_low, volume, fifty_two_week_high, fifty_two_week_low, market_cap, timestamp, previous_close, price_change, price_change_percent, dividend_yield, beta, next_earnings_date, error

## CSP Thresholds
- debt_to_equity: healthy < 1.0, warning 1.0-2.0, critical > 2.0
- current_ratio: healthy > 1.5, warning 1.0-1.5, critical < 1.0
- interest_coverage: healthy > 3.0, warning 1.5-3.0, critical < 1.5
- quick_ratio: healthy > 1.0, warning 0.5-1.0, critical < 0.5
- pe_ratio: healthy < 50, warning 50-100, critical > 100
- roe: healthy > 0.05, warning 0-0.05, critical < 0
- gross_margin: healthy > 0.20, warning 0-0.20, critical < 0
- net_profit_margin: healthy > 0.05, warning 0-0.05, critical < 0

## KB Rules (6 Horn clauses, two inference layers)
Layer 1 — ratio symptoms fire per-axis risks:
1. IF debt_to_equity_high      THEN leverage_risk
2. IF current_ratio_low        THEN liquidity_risk
3. IF interest_coverage_low    THEN solvency_risk

Layer 2 — per-axis risks chain into the review flag:
4. IF leverage_risk AND liquidity_risk  THEN high_risk_company
5. IF solvency_risk                     THEN high_risk_company
6. IF high_risk_company                 THEN flag_for_review

Verdict mapping:
- flag_for_review derived         → FAIL
- any rule fired (no flag)        → WARNING
- no rules fired                  → PASS

## Agno Agent Architecture
- DataAgent — fetches ratios and real-time price via tools
- AnalysisAgent — runs CSP constraint checks via tools
- ComplianceAgent — runs KB compliance rules via tools
- FinancialAnalysisTeam — coordinator, injects CSP verdict before LLM responds
- Tools: get_financial_ratios_tool, check_constraints_tool, check_compliance_tool, detect_anomalies_tool, get_realtime_price_tool

## Two-Ticker Comparison Mode
Triggered by: vs, versus, compare, between, which is better, or, and (with two tickers)
- Detects second ticker using _detect_second_ticker()
- Fetches both tickers live
- Renders side-by-side Live Market Data panels
- Shows separate HOLD/AVOID banners per ticker
- Text response formatting still rough — known limitation

## Known Issues / Future Work
- Two-ticker chat comparison: live data panels work, text formatting messy
- llama3.2 tool calling unreliable — workaround: pre-fetch + inject
- Benchmark harness functional but full 50-100 scenario sweep not run (time constraint)
- Earnings call sentiment (nomic-embed-text) not implemented (descoped per Kalathur Apr 27 guidance)
- Transfer pricing component not implemented (descoped per Kalathur Apr 27 guidance)

## How to Run
```bash
conda activate spring_2026
pip install -r requirements.txt
ollama serve && ollama pull llama3.2
streamlit run app.py                # web app (recommended demo surface)
python chat.py                      # terminal chatbot
python run_demo.py                  # classical demo, no Ollama needed
pytest -v -m "not network"          # 17 offline tests
pytest -v                           # 22 tests, needs internet
python -m src.evaluation.benchmark  # AAPL + MSFT 3-condition benchmark
```

## AI Tool Disclosure
Built with Claude Code as coding assistant. Lucky Upadhyay designed all architecture, specified all constraints, made all design decisions. Claude Code translated specs to Python and generated test scaffolding.
