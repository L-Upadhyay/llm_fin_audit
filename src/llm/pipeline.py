"""
Verified single-call pipeline — the default way to answer a question.

    facts (classical layer)  ->  one LLM call, facts in context, JSON out
        ->  verify every number against the facts
        ->  if wrong: one retry with the specific errors as feedback
        ->  if still wrong: correct the numbers in place, visibly
        ->  recommendation appended from the CSP verdict (never the LLM)

This replaces the three-agent Agno team as the default: a 3B model can't
reliably coordinate delegation, and a single call with the facts already
in context gives the verifier a stable, short answer to check. The team is
still available (src/llm/agno_agents.py) as a baseline for evaluation.

`run()` returns the same keys as FinancialAnalysisTeam.run plus
"verification", so the UIs can switch between the two.
"""

import json
import re

from src.data.facts import build_facts, prompt_view
from src.formatting import (
    format_comparison_block,
    format_live_quote_block,
    recommendation_for_verdict,
    recommendation_line,
)
from src.llm.providers import get_provider
from src.llm.routing import detect_second_ticker, is_price_question
from src.verification.verifier import apply_corrections, verify_text


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

SYSTEM_PROMPT = """You are a financial analysis assistant for an audit team.
Answer the user's question using ONLY the facts provided as JSON.

Rules:
- Every number you write must come from the facts. You may round to two decimals. Never use numbers from memory.
- roe, gross_margin and net_profit_margin are fractions: 0.2715 means 27.15%.
- If a value is null, say it is unavailable. Do not estimate it.
- csp_verdict is the official verdict for each ticker. Do not contradict it and do not invent your own buy/sell recommendation.
- If data_quality_issues or missing_required_ratios are present, mention them as caveats.
- Name the ticker symbol next to every number you state, especially when two tickers are compared.
- Be concise: markdown, under 180 words.

Respond with JSON: {"answer": "<markdown answer>"}"""


_PARTIAL_ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)


def _parse_answer(raw):
    """
    Pull the answer out of the model's JSON.

    Falls back to salvaging the "answer" string from truncated JSON (output
    cut off by the token cap), then to the raw text.
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data["answer"].strip()
    except (json.JSONDecodeError, TypeError):
        pass
    m = _PARTIAL_ANSWER_RE.search(raw or "")
    if m:
        try:
            return json.loads(f'"{m.group(1)}"').strip()
        except json.JSONDecodeError:
            return m.group(1).replace("\\n", "\n").strip()
    return (raw or "").strip()


def _collapse_repeats(text):
    """Drop exact-duplicate sentences — a safety net for degenerate loops."""
    # Split on sentence ends and line breaks, keeping the separators so
    # markdown survives. Very short lines ("---", "* N/A") may repeat.
    parts = re.split(r"((?<=[.!?])[ \t]+|\n+)", text)
    seen, kept = set(), []
    for i in range(0, len(parts), 2):
        sentence, sep = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
        key = " ".join(sentence.lower().split())
        if len(key) >= 12 and key in seen:
            continue
        seen.add(key)
        kept.append(sentence + sep)
    return "".join(kept).strip()


def _claim_rows(report):
    return [
        {
            "ticker": c.claim.ticker,
            "metric": c.claim.metric,
            "kind": c.claim.kind,
            "stated": c.claim.raw,
            "status": c.status,
            "actual": c.expected_text or c.actual,
        }
        for c in report.checked
    ]


class AuditPipeline:
    """
    Parameters:
        provider      LLM provider (see providers.py); default from LLM_PROVIDER.
        verify        run the claim verifier (False = unverified baseline).
        max_retries   rewrite attempts after a failed verification.
        correct       rewrite remaining wrong numbers in place.
        fact_source   callable(ticker) -> facts dict; default live yfinance.
                      The evaluation passes a snapshot for reproducibility.
    """

    def __init__(self, provider=None, verify=True, max_retries=1, correct=True,
                 fact_source=build_facts):
        self.provider = provider or get_provider()
        self.verify = verify
        self.max_retries = max_retries
        self.correct = correct
        self.fact_source = fact_source

    def _ask(self, user):
        raw = self.provider.complete(SYSTEM_PROMPT, user, ANSWER_SCHEMA)
        return _collapse_repeats(_parse_answer(raw))

    def run(self, ticker, question):
        ticker = ticker.upper()
        second = detect_second_ticker(question, ticker)
        tickers = [ticker] + ([second] if second else [])
        facts = {t: self.fact_source(t) for t in tickers}

        user = (
            f"Question: {question}\n"
            f"Primary ticker: {ticker}\n\n"
            f"Facts (JSON):\n"
            f"{json.dumps([prompt_view(facts[t]) for t in tickers], indent=1, default=str)}"
        )

        answer = self._ask(user)
        first_answer = answer
        report = verify_text(answer, facts, ticker)
        first_summary = report.summary()
        retries = 0
        while self.verify and report.contradicted and retries < self.max_retries:
            retry_user = (
                f"{user}\n\nYour previous answer was:\n{answer}\n\n"
                f"It contains these errors:\n{report.feedback()}\n\n"
                "Rewrite the full answer with the errors fixed. "
                "Use only numbers that appear in the facts."
            )
            answer = self._ask(retry_user)
            report = verify_text(answer, facts, ticker)
            retries += 1

        uncorrected_answer = answer
        corrections = []
        if self.verify and self.correct and report.contradicted:
            answer, corrections = apply_corrections(answer, report)

        verification = None
        if self.verify:
            verification = {
                "first_attempt": first_summary,
                "final": report.summary(),
                "retries": retries,
                "corrections": corrections,
                "claims": _claim_rows(report),
                # Drafts kept for auditing and re-scoring in evaluation.
                "first_answer": first_answer,
                "uncorrected_answer": uncorrected_answer,
            }

        return self._compose(ticker, question, tickers, facts, answer, verification)

    def _compose(self, ticker, question, tickers, facts, answer, verification):
        """Wrap the answer with the same blocks and keys as the agent team."""
        blocks = [
            {
                "ticker": t,
                "ratios": facts[t]["ratios"],
                "realtime": {**facts[t]["realtime"], "ticker": t,
                             "timestamp": facts[t].get("quote_time")},
                "csp_verdict": facts[t]["csp_verdict"],
                "recommendation": recommendation_for_verdict(facts[t]["csp_verdict"]),
            }
            for t in tickers
        ]
        primary = blocks[0]
        comparison = blocks if len(blocks) > 1 else None
        realtime = primary["realtime"] if (is_price_question(question) and not comparison) else None

        text = answer
        if comparison:
            text = f"{format_comparison_block(comparison)}\n\n{text}"
            rec = "\n".join(f"- **{b['ticker']}**: {recommendation_line(b['recommendation'])}"
                            for b in comparison)
        else:
            if realtime:
                quote_block = format_live_quote_block(realtime)
                if quote_block:
                    text = f"{quote_block}\n\n{text}"
            rec = recommendation_line(primary["recommendation"])
        text = f"{text.rstrip()}\n\n## Recommendation\n{rec}"

        return {
            "text": text,
            "answer": answer,
            "csp_verdict": primary["csp_verdict"],
            "recommendation": primary["recommendation"],
            "ratios": primary["ratios"],
            "realtime": realtime,
            "comparison": comparison,
            "verification": verification,
            "facts": facts,
        }
