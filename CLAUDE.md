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

Three layers:
1. Classical layer (deterministic) — CSP solver, KB forward chaining, anomaly detector, consistency constraints
2. LLM layer — verified single-call pipeline (default) or legacy Agno multi-agent team; verdict always from the CSP
3. Verification layer — extracts every number/verdict label from the LLM's text, checks it against the facts,
   triggers one rewrite with feedback, then corrects remaining errors in place ("0.89 [corrected from 1.43]")

## File Structure
src/
data/loader.py          — yfinance ratio extraction (8 ratios) + get_realtime_price() (16 fields)
classical/
csp_solver.py         — Variable, Constraint, FinancialCSP with AC-3 + backtracking + forward checking
knowledge_base.py     — Horn-clause Clause/KnowledgeBase with forward chaining, 6 compliance rules
anomaly_detector.py   — robust (median/MAD) z-score outlier flagging, worst-quarter lookup, trend classification
thresholds.py         — single source of truth for ratio cut-offs, REQUIRED_RATIOS, verdict labels
comparator.py         — multi-stock comparison, composite risk ranking, matplotlib charts
llm/
agno_agents.py        — DataAgent, AnalysisAgent, ComplianceAgent, FinancialAnalysisTeam
evaluation/
benchmark.py          — 3-condition evaluation harness (classical-only, LLM-only, hybrid)
consistency.py        — accounting-identity constraints on source data (quick <= current, net <= gross margin, ...)
data/facts.py           — build_facts(ticker): everything the classical layer knows; the LLM's only input
llm/
pipeline.py           — AuditPipeline: facts -> 1 LLM call (JSON) -> verify -> 1 rewrite -> correct
providers.py          — ollama:<model>, openai:<model> (OpenAI-compatible); LLM_PROVIDER env var
routing.py            — price-question and second-ticker detection
verification/
claims.py             — rule-based claim extraction (prose, tables, heading+bullet, comparisons, verdicts)
verifier.py           — rounding-aware checks, feedback text, in-place corrections
evaluation/
snapshot.py           — freezes facts for 31 tickers -> data/fixtures/facts_snapshot.json
injection.py          — injected-error precision/recall of the verifier (offline, in CI)
live_eval.py          — llm_only vs team vs pipeline on real model output -> results/
formatting.py             — shared display helpers (price/market-cap formatting, response cleaning, recommendation labels)
tests/                    — 43 pytest tests (38 offline, 5 network)
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
- Default engine is AuditPipeline; FinancialAnalysisTeam is kept as a legacy baseline for evaluation
- Every LLM call goes through a provider with MAX_OUTPUT_TOKENS and REQUEST_TIMEOUT_S (llama3.2 can loop at temperature 0)
- Never let unverified LLM numbers reach the user from the pipeline; the verifier must stay deterministic (no LLM judge)
- LLM cannot output financial verdict without CSP solver verifying first
- HOLD/WATCH/AVOID recommendation ALWAYS driven by CSP verdict, never LLM opinion
- HOLD = CSP PASS (green), WATCH = CSP WARNING (yellow), AVOID = CSP FAIL (red), NO VERDICT = INSUFFICIENT_DATA
- Real-time price ALWAYS pre-fetched from yfinance and injected — never let LLM answer price questions from training data
- Strip all raw JSON delegation text (delegate_task_to_member, DataAgent:, AnalysisAgent:, etc.) from responses before displaying

### Tests
- Always run pytest after any change
- Never break existing tests to add new features
- Any test that touches yfinance or Ollama MUST be marked @pytest.mark.network
- Offline run: pytest -v -m "not network" (this is what CI runs, plus ruff)
- Full run:    pytest -v, requires internet
- Any verifier bug found in real output gets a unit test in tests/test_verifier.py

### Git
- Commit after every working feature
- Push to origin main
- Clear descriptive commit messages

## Financial Ratios (8 total)
1. debt_to_equity — Total Debt / Stockholders Equity, latest annual balance sheet
2. current_ratio — Current Assets / Current Liabilities, latest annual balance sheet
3. interest_coverage_ratio — EBIT / |Interest Expense|, latest annual income statement
4. quick_ratio — from ticker.info["quickRatio"]
5. pe_ratio — from ticker.info["trailingPE"]
6. roe — from ticker.info["returnOnEquity"]
7. gross_margin — from ticker.info["grossMargins"]
8. net_profit_margin — from ticker.info["profitMargins"]

## Live Market Data Fields (16 total)
current_price, open, day_high, day_low, volume, fifty_two_week_high, fifty_two_week_low, market_cap, timestamp, previous_close, price_change, price_change_percent, dividend_yield, beta, next_earnings_date, error

## CSP Thresholds (defined in src/classical/thresholds.py — change them there only)
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

Base facts fire when the ratio is in its critical band (thresholds.py).

Verdict mapping (CSP and KB):
- debt_to_equity or current_ratio missing → INSUFFICIENT_DATA (fail closed, never PASS)
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
Triggered by: vs, versus, compare, between, which is better, or, and — plus a second ticker typed in UPPERCASE or as a $cashtag
- Detects second ticker using _detect_second_ticker()
- Fetches both tickers live
- Renders side-by-side Live Market Data panels
- Shows separate HOLD/AVOID banners per ticker
- Text response formatting still rough — known limitation

## Known Issues / Future Work
- Two-ticker chat comparison: live data panels work, text formatting messy
- llama3.2 tool calling unreliable — the legacy team often returns delegation chatter with no numbers
- Claim extraction misses paraphrased metrics and value-before-name phrasing (0% on the "hard" injection family)
- Qualitative claims ("MSFT is critical", "healthy") are not verified — only numbers and verdict labels
- Earnings call sentiment (nomic-embed-text) not implemented (descoped per Kalathur Apr 27 guidance)
- Transfer pricing component not implemented (descoped per Kalathur Apr 27 guidance)

## How to Run
```bash
conda activate spring_2026   # or: python -m venv .venv
pip install -r requirements.txt
ollama serve && ollama pull llama3.2
streamlit run app.py                # web app (recommended demo surface)
python chat.py                      # terminal chatbot
python run_demo.py                  # classical demo, no Ollama needed
pytest -v -m "not network"          # offline tests (CI)
python -m src.evaluation.injection  # verifier precision/recall on injected errors
python -m src.evaluation.live_eval  # llm_only vs team vs pipeline (needs Ollama, ~35 min)
python -m src.evaluation.snapshot   # refresh the facts snapshot (changes eval ground truth)
```

## AI Tool Disclosure
Built with Claude Code as coding assistant. Lucky Upadhyay designed all architecture, specified all constraints, made all design decisions. Claude Code translated specs to Python and generated test scaffolding.
