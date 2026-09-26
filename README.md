# llm_fin_audit

[![CI](https://github.com/L-Upadhyay/llm_fin_audit/actions/workflows/ci.yml/badge.svg)](https://github.com/L-Upadhyay/llm_fin_audit/actions/workflows/ci.yml)

A deterministic audit layer that sits between an LLM and anyone relying on its financial analysis. Classical AI (constraint satisfaction, forward-chaining rules, robust anomaly detection) computes the facts and owns the verdict. The LLM writes the explanation. A rule-based verifier then checks every number and label in that explanation against the facts before anyone reads it.

## Why

Asked whether Apple was financially healthy, llama3.2 with no tools answered:

> Debt-to-Equity: **0.12** … Current Ratio: **1.43** … Apple appears financially healthy.

The actual figures at the time, from the annual balance sheet data the system pulls, were **1.34** and **0.89**. A current ratio below 1.0 means short-term liabilities exceed short-term assets. The model was fluent, specific, and wrong. For an equity analyst or an auditor, that is the failure that matters: the model gives confident numbers with nothing to trace them back to.

## Results

All numbers below are reproducible from the repo: facts are frozen in `data/fixtures/facts_snapshot.json` (31 tickers), and every model answer is in `results/`.

**Live evaluation: llama3.2 (3B, local), 31 tickers × 3 questions = 93 answers per condition.** Every number and verdict label in every answer is checked against the snapshot. Brackets are Wilson 95% intervals.

| Condition | Wrong numbers / labels | Answers with ≥1 error | Answers with no checkable numbers | Median latency |
|---|---|---|---|---|
| LLM alone (no data) | 97.1% (94–99%) | 78.5% | 21.5% | 6.5 s |
| Original Agno agent team (tools, unverified) | 50.5% (41–60%) | 14.0% | **72.0%** | 11.0 s |
| Verified pipeline, first draft | 3.3% (2–6%) | 12.9% | 1.1% | 4.0 s |
| Verified pipeline, after one rewrite with the verifier's feedback | 1.0% (0–3%) | 4.3% | 1.1% | 4.0 s |
| Verified pipeline, delivered (re-verified after in-place correction) | 0.0% (0–1%) | 0.0% | 1.1% | 4.0 s |

What this shows:

- **Grounding does most of the work, and the verifier catches what grounding misses.** The same model goes from 97% wrong numbers without data to 3.3% with the facts in context. The rewrite cuts that to 1.0%. The last 4 errors are corrected in place, visibly.
- **The multi-agent team was the weakest design.** Three agents on a 3B model mostly returned delegation chatter: 72% of its answers contained no checkable number at all, and half of the numbers it did state were wrong. That is why it's no longer the default.
- **Prompt formatting mattered.** In the first run, 22 of 39 first-draft errors were unit slips (ROE 0.1223 written as "0.1223%"). Giving the model pre-formatted values ("12.23%", "$3.83T") cut the first-draft error rate from 9.1% to 3.3%.
- **"Delivered 0.0%" means the verifier finds nothing left, not that the answer is perfect.** The verifier can only judge what it extracts: 0.56 numbers per pipeline answer went unextracted. Its blind spots are measured separately below.

**How much to trust the verifier.**

- *Injected errors* (`results/injection_eval.md`): 676 known errors planted in 248 generated narratives. Precision was 100% with 0 false alarms on 1,155 clean statements. Recall was 99.7% on the phrasings the extractor targets and **0% on phrasings it wasn't built for** (value before the metric name, paraphrases like "current assets cover current liabilities 0.89 times").
- *Audit of real output.* I read every error the verifier flagged in llama3.2's drafts. First run: 39 of 42 flags were genuine. Second run: 13 of 17. Each of the 7 false alarms (verdict labels referenced after the label, one truncated value, a percentage paired with a dollar metric) became a regression test and was fixed. One of them had caused a harmful "correction", which is why corrections are always marked in the text. In the final run all 13 flags are genuine, but those rules were tuned on these same answers, so treat that as a lower bound on future false alarms, not proof.

## How it works

```
 yfinance ─► facts (classical layer, deterministic)
             ├─ 8 ratios + live quote + EPS history
             ├─ CSP solver ............ PASS / WARNING / FAIL / INSUFFICIENT_DATA
             ├─ Horn-clause KB ........ compliance rules fired
             ├─ anomaly detector ...... robust z-score on 8 quarters of EPS
             └─ consistency checks .... quick ≤ current, net ≤ gross margin, …
                       │
                       ▼
             one LLM call, facts in context, JSON out
                       │
                       ▼
             claim verifier ── every number / verdict label → checked against the facts
                       │  wrong?  → one rewrite with the specific errors as feedback
                       │  still wrong? → corrected in place: "0.89 [corrected from 1.43]"
                       ▼
             answer + verification report + recommendation (from the CSP, never the LLM)
```

- **The verdict never comes from the LLM.** `FinancialCSP` computes it before the model runs, and the HOLD / WATCH / AVOID banner is derived from it.
- **Missing data fails closed.** If leverage or liquidity ratios are unavailable (unknown ticker, rate limit, banks without a current ratio), the answer is `INSUFFICIENT_DATA`, never a default PASS.
- **The verifier is deterministic.** No second LLM grades the first. Claims are extracted by rules covering prose, markdown tables, heading-then-bullet layouts, comparisons ("below 1.0"), verdict labels, and per-ratio status labels. They are matched to a ticker and checked with the rounding the model itself used.
- **Corrections are visible.** A reader sees exactly which numbers were changed and from what.

## Components

| Module | What it does |
|---|---|
| `src/classical/thresholds.py` | Single source of truth for ratio cut-offs and verdict labels |
| `src/classical/csp_solver.py` | CSP from scratch: AC-3 arc consistency, backtracking with forward checking |
| `src/classical/knowledge_base.py` | Horn-clause KB with forward chaining; leverage/liquidity/solvency risks chain to `flag_for_review` |
| `src/classical/anomaly_detector.py` | Modified z-score (median/MAD) outlier detection on quarterly EPS |
| `src/classical/consistency.py` | Accounting-identity constraints that flag inconsistent source data |
| `src/data/facts.py` | Everything the classical layer knows about a ticker; the LLM's only input |
| `src/verification/` | Claim extraction and verification, feedback text, in-place corrections |
| `src/llm/pipeline.py` | The verified single-call pipeline (default engine) |
| `src/llm/providers.py` | `ollama:<model>` or any OpenAI-compatible endpoint (`openai:<model>`) |
| `src/llm/agno_agents.py` | The original three-agent Agno team, kept as a baseline |
| `src/evaluation/` | Facts snapshot, injected-error suite, live evaluation |

## Run it

```bash
git clone https://github.com/L-Upadhyay/llm_fin_audit
cd llm_fin_audit
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
ollama pull llama3.2        # https://ollama.com — needed for chat and the live eval
```

| Interface | Command |
|---|---|
| Web app (Analysis / Compare / Chat; pick the chat engine in the sidebar) | `streamlit run app.py` |
| Terminal chatbot (verified pipeline; `--engine team` for the Agno team) | `python chat.py` |
| Classical-only demo, no LLM | `python run_demo.py MSFT` |
| Offline tests (what CI runs) | `pytest -m "not network"` |
| Injected-error evaluation (offline) | `python -m src.evaluation.injection` |
| Live evaluation (~40 min on an M4; keep the machine awake) | `caffeinate -i python -m src.evaluation.live_eval` |

To use another model, set `LLM_PROVIDER`, e.g. `LLM_PROVIDER=openai:gpt-4o-mini` with `OPENAI_API_KEY`, or point `OPENAI_BASE_URL` at vLLM, Groq, or Together.

## Limitations

- **The extractor only knows the phrasings it was built for.** Paraphrases ("current assets cover current liabilities 0.89 times"), a value placed before the metric name, and long-distance references are not extracted (0% recall on those injected errors). Unextracted numbers are counted as "untracked" in every report rather than silently passed.
- **Qualitative claims are not verified.** "Profitability is strong" or "both companies are critical" pass through. Only numbers and verdict labels are checked.
- **One model so far.** The live evaluation uses llama3.2 (3B) locally. Larger models will hallucinate less, and whether the verifier still earns its latency there is the next experiment.
- **Thresholds are generic, not sector-aware.** One set of cut-offs is applied to every industry, so Walmart's and Apple's thin working capital reads as critical. That is why most of the snapshot is FAIL.
- **yfinance is a scraper**, not an authoritative filings source, and it mixes the latest annual balance sheet with trailing-twelve-month figures. The consistency checks catch some of this, and SEC EDGAR XBRL would fix it.
- **Not investment advice.** HOLD / WATCH / AVOID are labels for ratio-based risk bands.

## Background

This started as my term project for BU MET CS 664 (Artificial Intelligence, Spring 2026) and I'm continuing to develop it. It draws on five years of financial analysis at PwC and EY. Built with Claude Code as a coding assistant.

## License

MIT
