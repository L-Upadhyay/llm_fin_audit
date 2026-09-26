"""
Claim verification: reconcile extracted claims against the facts the
classical layer computed, and optionally correct the text.

Facts shape (one entry per ticker):
    {
      "AAPL": {"ratios": {...}, "realtime": {...}, "csp_verdict": "FAIL"},
      ...
    }

Statuses:
    supported     the claim matches the fact within rounding tolerance
    contradicted  the claim disagrees with the fact
    unverifiable  the fact is missing (None) or the ticker isn't covered
"""

from dataclasses import dataclass

from src.verification.claims import METRICS_BY_KEY, extract_claims, untracked_numbers


SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNVERIFIABLE = "unverifiable"

# Relative slack on top of rounding: absorbs rounding on scaled values
# ("$3.9T") and small differences between yfinance refreshes.
REL_TOLERANCE = 0.01


@dataclass
class CheckedClaim:
    claim: object
    status: str
    actual: object = None       # fact value (float) or verdict string
    expected_text: str = ""     # how the correct value should read


def _fact_for(claim, facts):
    ticker_facts = facts.get(claim.ticker)
    if ticker_facts is None:
        return None
    if claim.kind == "verdict":
        return ticker_facts.get("csp_verdict")
    spec = METRICS_BY_KEY[claim.metric]
    return (ticker_facts.get(spec.source) or {}).get(claim.metric)


def _comparable(claim, actual):
    """
    Express the fact in the same units the claim was written in.

    Returns (stated, actual_in_claim_units, tolerance).
    """
    spec = METRICS_BY_KEY[claim.metric]
    n = claim.number
    stated = n.value
    a = actual

    if spec.kind == "fraction":
        # Stored 0.2715; "27.15%" or "27.15" -> percent units, "0.27" -> fraction.
        if n.is_percent or abs(stated) > 1.5:
            a = actual * 100
    elif spec.kind == "multiple" and n.is_percent:
        # "D/E of 134%" is a legitimate way to write 1.34.
        a = actual * 100
    # "percent" (dividend yield) and "currency" compare as written.

    rounding = 0.5 * 10 ** (-n.decimals) * n.scale
    tol = max(rounding, abs(a) * REL_TOLERANCE)
    return stated, a, tol


def _format_like(claim, actual):
    """Render the correct value in the claim's own style."""
    spec = METRICS_BY_KEY[claim.metric]
    n = claim.number
    decimals = max(n.decimals, 2)
    if spec.kind == "currency":
        if n.scale > 1:
            suffix = {1e12: "T", 1e9: "B", 1e6: "M", 1e3: "K"}.get(n.scale, "")
            return f"${actual / n.scale:,.{decimals}f}{suffix}"
        return f"${actual:,.{decimals}f}"
    if spec.kind == "fraction":
        if n.is_percent or abs(n.value) > 1.5:
            return f"{actual * 100:.{decimals}f}%"
        return f"{actual:.{decimals}f}"
    if spec.kind == "percent":
        return f"{actual:.{decimals}f}%"
    if n.is_percent:
        return f"{actual * 100:.{decimals}f}%"
    return f"{actual:.{decimals}f}"


def check_claim(claim, facts):
    actual = _fact_for(claim, facts)
    if actual is None:
        return CheckedClaim(claim, UNVERIFIABLE)

    if claim.kind == "verdict":
        ok = claim.verdict == actual
        return CheckedClaim(claim, SUPPORTED if ok else CONTRADICTED, actual, actual)

    stated, a, tol = _comparable(claim, actual)
    if claim.kind == "comparison":
        ok = a < stated + tol if claim.comparator == "lt" else a > stated - tol
    else:
        ok = abs(stated - a) <= tol
    return CheckedClaim(claim, SUPPORTED if ok else CONTRADICTED, actual,
                        _format_like(claim, actual))


@dataclass
class VerificationReport:
    checked: list
    untracked: list

    def by_status(self, status):
        return [c for c in self.checked if c.status == status]

    @property
    def contradicted(self):
        return self.by_status(CONTRADICTED)

    @property
    def supported(self):
        return self.by_status(SUPPORTED)

    @property
    def unverifiable(self):
        return self.by_status(UNVERIFIABLE)

    def summary(self):
        return {
            "claims": len(self.checked),
            "supported": len(self.supported),
            "contradicted": len(self.contradicted),
            "unverifiable": len(self.unverifiable),
            "untracked_numbers": len(self.untracked),
        }

    def feedback(self):
        """Plain-English list of errors, used to ask the model for a rewrite."""
        lines = []
        for c in self.contradicted:
            cl = c.claim
            label = "verdict" if cl.kind == "verdict" else METRICS_BY_KEY[cl.metric].label
            if cl.kind == "comparison":
                word = "below" if cl.comparator == "lt" else "above"
                lines.append(f"- {cl.ticker} {label}: you said it is {word} {cl.raw}, "
                             f"but the actual value is {c.expected_text}.")
            else:
                lines.append(f"- {cl.ticker} {label}: you wrote {cl.raw}, "
                             f"the correct value is {c.expected_text}.")
        return "\n".join(lines)


def verify_text(text, facts, primary):
    """Extract and check every claim in `text`."""
    claims = extract_claims(text, list(facts.keys()), primary)
    checked = [check_claim(c, facts) for c in claims]
    return VerificationReport(checked, untracked_numbers(text, claims))


def apply_corrections(text, report):
    """
    Rewrite contradicted claims in place so the reader never sees an
    unflagged wrong number:

      value / verdict   "0.12"          -> "1.34 [corrected from 0.12]"
      comparison        "below 1.0"     -> "below 1.0 [unverified: actual 1.35]"

    Returns (new_text, corrections) where corrections is a list of dicts.
    """
    corrections = []
    # Apply right-to-left so earlier spans stay valid.
    for c in sorted(report.contradicted, key=lambda c: c.claim.start, reverse=True):
        cl = c.claim
        if cl.kind == "comparison":
            replacement = f"{cl.raw} [unverified: actual {c.expected_text}]"
        else:
            replacement = f"{c.expected_text} [corrected from {cl.raw}]"
        text = text[:cl.start] + replacement + text[cl.end:]
        corrections.append({
            "ticker": cl.ticker,
            "metric": cl.metric,
            "kind": cl.kind,
            "stated": cl.raw,
            "actual": c.expected_text,
        })
    corrections.reverse()
    return text, corrections
