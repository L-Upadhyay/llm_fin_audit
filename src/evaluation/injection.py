"""
Injected-error evaluation of the claim verifier (offline, deterministic).

Builds synthetic analyst-style narratives from the facts snapshot, plants
known errors in some of the numbers, and measures whether the verifier
flags exactly those.

    python -m src.evaluation.injection      # writes results/injection_eval.{json,md}

Every numeric statement is a *slot* with a known span, metric, ticker, and
label (clean or injected, and which error type). A slot counts as:
    flagged    if any verifier claim overlapping its span is contradicted
    extracted  if any verifier claim overlaps its span at all

Two template families:
    standard   phrasings the extractor was designed for (prose, tables,
               heading + bullet, two-ticker sentences, verdicts, comparisons)
    hard       phrasings it was NOT designed for (value before the metric
               name, paraphrased metrics, long-distance references). These
               measure recall honestly outside the happy path.

Caveat: the templates were written alongside the extractor, so "standard"
recall is an upper bound. The live evaluation (live_eval.py) measures the
verifier on real model output.
"""

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass

from src.classical.thresholds import RATIO_THRESHOLDS
from src.evaluation.snapshot import load_snapshot
from src.verification.verifier import CONTRADICTED, verify_text


SEED = 664
DOCS_PER_TICKER = 8
INJECT_PROB = 0.35
RESULTS_DIR = "results"

VERDICTS = ("PASS", "WARNING", "FAIL")


# ---------------------------------------------------------------------- #
# Rendering values the way an analyst (or an LLM) would write them
# ---------------------------------------------------------------------- #

MULTIPLES = ("debt_to_equity", "current_ratio", "quick_ratio",
             "interest_coverage_ratio", "pe_ratio")
FRACTIONS = ("roe", "gross_margin", "net_profit_margin")

LABELS = {
    "debt_to_equity": ("debt-to-equity ratio", "D/E ratio", "debt to equity"),
    "current_ratio": ("current ratio",),
    "quick_ratio": ("quick ratio", "acid-test ratio"),
    "interest_coverage_ratio": ("interest coverage", "interest coverage ratio"),
    "pe_ratio": ("P/E ratio", "price-to-earnings ratio", "P/E"),
    "roe": ("ROE", "return on equity"),
    "gross_margin": ("gross margin", "gross profit margin"),
    "net_profit_margin": ("net profit margin", "net margin", "profit margin"),
    "current_price": ("share price", "current price"),
    "market_cap": ("market cap", "market capitalization"),
}


def render(metric, value, rng, style=None):
    """Format a value in one of several realistic styles."""
    if metric in MULTIPLES:
        d = rng.choice((1, 2, 2, 3)) if style is None else style
        suffix = "x" if metric in ("interest_coverage_ratio", "debt_to_equity") and rng.random() < 0.3 else ""
        return f"{value:.{d}f}{suffix}"
    if metric in FRACTIONS:
        if style == "fraction" or (style is None and rng.random() < 0.2):
            return f"{value:.3f}"
        d = rng.choice((1, 2))
        return f"{value * 100:.{d}f}%"
    if metric == "current_price":
        return f"${value:,.2f}"
    if metric == "market_cap":
        if value >= 1e12 and rng.random() < 0.5:
            return f"${value / 1e12:.2f} trillion"
        return f"${value / 1e9:,.1f}B"
    raise KeyError(metric)


def _differs(a, b, rel=0.10):
    return abs(a - b) > max(abs(b) * rel, 0.02)


def corrupt(kind, metric, value, rng, ticker_facts, peer_facts):
    """Return a wrong value for `kind`, or None if not applicable."""
    if kind == "scale":
        return value * rng.choice((0.4, 0.5, 0.6, 1.6, 2.0, 3.0))
    if kind == "near_miss":
        wrong = value * (1 + rng.choice((-1, 1)) * rng.uniform(0.10, 0.20))
        return wrong if _differs(wrong, value) else None
    if kind == "sign_flip":
        return -value if abs(value) > 0.05 else None
    if kind == "metric_swap":
        pool = [v for m, v in ticker_facts["ratios"].items()
                if m != metric and v is not None and _same_family(m, metric) and _differs(v, value)]
        return rng.choice(pool) if pool else None
    if kind == "ticker_swap":
        other = _value(peer_facts, metric) if peer_facts else None
        return other if other is not None and _differs(other, value) else None
    raise KeyError(kind)


def _same_family(a, b):
    return (a in MULTIPLES) == (b in MULTIPLES)


def _value(facts, metric):
    if metric in ("current_price", "market_cap"):
        return facts["realtime"].get(metric)
    return facts["ratios"].get(metric)


NUMERIC_ERRORS = ("scale", "near_miss", "sign_flip", "metric_swap", "ticker_swap")


# ---------------------------------------------------------------------- #
# Documents with tracked slots
# ---------------------------------------------------------------------- #

@dataclass
class Slot:
    start: int
    end: int
    metric: str
    ticker: str
    family: str
    template: str
    error: str = None      # None = clean


class Doc:
    def __init__(self):
        self.parts, self.length, self.slots = [], 0, []

    def add(self, s):
        self.parts.append(s)
        self.length += len(s)

    def slot(self, s, **kw):
        self.slots.append(Slot(self.length, self.length + len(s), **kw))
        self.add(s)

    @property
    def text(self):
        return "".join(self.parts)


class Builder:
    """Renders one templated statement, deciding per slot whether to inject."""

    def __init__(self, rng, facts, ticker, peer):
        self.rng, self.facts, self.ticker, self.peer = rng, facts, ticker, peer

    def value_slot(self, doc, metric, ticker, family, template, allow=NUMERIC_ERRORS,
                   style=None):
        true = _value(self.facts[ticker], metric)
        error = None
        shown = true
        if self.rng.random() < INJECT_PROB:
            kinds = [k for k in allow if not (k == "ticker_swap" and ticker != self.ticker)]
            self.rng.shuffle(kinds)
            for kind in kinds:
                peer_facts = self.facts.get(self.peer) if kind == "ticker_swap" else None
                wrong = corrupt(kind, metric, true, self.rng, self.facts[ticker], peer_facts)
                if wrong is not None:
                    error, shown = kind, wrong
                    break
        text = render(metric, shown, self.rng, style)
        if error is None and metric in FRACTIONS and self.rng.random() < 0.12:
            # Unit confusion: a fraction written as if it were a percent.
            text, error = f"{true:.2f}%", "unit_confusion"
        doc.slot(text, metric=metric, ticker=ticker, family=family, template=template,
                 error=error)


def available(facts, ticker, metrics):
    return [m for m in metrics if _value(facts[ticker], m) is not None]


# Each template: (name, family, needs_peer, fn(builder, doc)) -> bool (False = skip)

def t_prose_is(b, doc):
    ms = available(b.facts, b.ticker, list(LABELS))
    if not ms:
        return False
    m = b.rng.choice(ms)
    doc.add(f"{b.ticker}'s {b.rng.choice(LABELS[m])} is ")
    b.value_slot(doc, m, b.ticker, "standard", "prose_is")
    doc.add(". ")
    return True


def t_prose_of(b, doc):
    ms = available(b.facts, b.ticker, MULTIPLES + FRACTIONS)
    if len(ms) < 2:
        return False
    m1, m2 = b.rng.sample(ms, 2)
    doc.add(f"{b.ticker} reports a {b.rng.choice(LABELS[m1])} of ")
    b.value_slot(doc, m1, b.ticker, "standard", "prose_of")
    doc.add(f", alongside a {b.rng.choice(LABELS[m2])} of ")
    b.value_slot(doc, m2, b.ticker, "standard", "prose_of")
    doc.add(". ")
    return True


def t_table(b, doc):
    ms = available(b.facts, b.ticker, MULTIPLES + FRACTIONS)
    if len(ms) < 2:
        return False
    doc.add(f"\n| Metric | {b.ticker} |\n|---|---|\n")
    for m in b.rng.sample(ms, min(3, len(ms))):
        doc.add(f"| {LABELS[m][0].title()} | ")
        b.value_slot(doc, m, b.ticker, "standard", "table")
        doc.add(" |\n")
    doc.add("\n")
    return True


def t_heading_bullet(b, doc):
    ms = available(b.facts, b.ticker, MULTIPLES + FRACTIONS)
    if not ms:
        return False
    m = b.rng.choice(ms)
    doc.add(f"\n**{LABELS[m][0].title()}**\n* {b.ticker}: ")
    b.value_slot(doc, m, b.ticker, "standard", "heading_bullet")
    doc.add("\n\n")
    return True


def t_two_ticker(b, doc):
    if not b.peer:
        return False
    ms = [m for m in available(b.facts, b.ticker, MULTIPLES + FRACTIONS)
          if _value(b.facts[b.peer], m) is not None]
    if not ms:
        return False
    m = b.rng.choice(ms)
    label = b.rng.choice(LABELS[m])
    doc.add(f"{b.ticker} has a {label} of ")
    b.value_slot(doc, m, b.ticker, "standard", "two_ticker")
    doc.add(f", while {b.peer} has a {label} of ")
    b.value_slot(doc, m, b.peer, "standard", "two_ticker",
                 allow=("scale", "near_miss", "sign_flip", "metric_swap"))
    doc.add(". ")
    return True


def t_verdict(b, doc):
    true = b.facts[b.ticker]["csp_verdict"]
    if true not in VERDICTS:
        return False
    shown, error = true, None
    if b.rng.random() < INJECT_PROB:
        shown, error = b.rng.choice([v for v in VERDICTS if v != true]), "verdict_flip"
    doc.add(f"The CSP verdict for {b.ticker} is ")
    doc.slot(shown, metric="verdict", ticker=b.ticker, family="standard",
             template="verdict", error=error)
    doc.add(". ")
    return True


def t_comparison(b, doc):
    ms = [m for m in available(b.facts, b.ticker, list(RATIO_THRESHOLDS))]
    b.rng.shuffle(ms)
    for m in ms:
        value = b.facts[b.ticker]["ratios"][m]
        cut = RATIO_THRESHOLDS[m]["critical"]
        scale = 100 if m in FRACTIONS else 1
        if abs(value - cut) <= max(abs(cut) * 0.05, 0.01):
            continue  # too close to call
        truly_below = value < cut
        below = truly_below
        error = None
        if b.rng.random() < INJECT_PROB:
            below, error = not truly_below, "comparison_flip"
        word = "below" if below else "above"
        cut_text = f"{cut * scale:g}%" if m in FRACTIONS else f"{cut:g}"
        doc.add(f"{b.ticker}'s {LABELS[m][0]} is {word} ")
        doc.slot(cut_text, metric=m, ticker=b.ticker, family="standard",
                 template="comparison", error=error)
        doc.add(". ")
        return True
    return False


# --- "hard": phrasings the extractor was not designed for ----------------

def t_value_first(b, doc):
    ms = available(b.facts, b.ticker, MULTIPLES)
    if not ms:
        return False
    m = b.rng.choice(ms)
    doc.add("At ")
    b.value_slot(doc, m, b.ticker, "hard", "value_first")
    doc.add(f", {b.ticker}'s {LABELS[m][0]} stands out. ")
    return True


def t_paraphrase(b, doc):
    options = []
    if _value(b.facts[b.ticker], "current_ratio") is not None:
        options.append(("current_ratio", "current assets cover current liabilities ", " times"))
    if _value(b.facts[b.ticker], "debt_to_equity") is not None:
        options.append(("debt_to_equity", "the company carries ", " dollars of debt per dollar of equity"))
    if _value(b.facts[b.ticker], "roe") is not None:
        options.append(("roe", "shareholders earn a return of ", " on their equity"))
    if not options:
        return False
    m, before, after = b.rng.choice(options)
    doc.add(f"For {b.ticker}, {before}")
    b.value_slot(doc, m, b.ticker, "hard", "paraphrase")
    doc.add(f"{after}. ")
    return True


def t_long_distance(b, doc):
    ms = available(b.facts, b.ticker, MULTIPLES)
    if not ms:
        return False
    m = b.rng.choice(ms)
    doc.add(f"{b.ticker}'s {LABELS[m][0]}, which management has repeatedly highlighted on "
            f"earnings calls as a priority given the tighter credit conditions this cycle, "
            f"came in at ")
    b.value_slot(doc, m, b.ticker, "hard", "long_distance")
    doc.add(". ")
    return True


STANDARD = (t_prose_is, t_prose_of, t_table, t_heading_bullet, t_two_ticker,
            t_verdict, t_comparison)
HARD = (t_value_first, t_paraphrase, t_long_distance)


# ---------------------------------------------------------------------- #
# Scoring
# ---------------------------------------------------------------------- #

def score_doc(doc, facts, primary):
    tickers = sorted({s.ticker for s in doc.slots} | {primary})
    report = verify_text(doc.text, {t: facts[t] for t in tickers}, primary)
    rows = []
    matched = set()
    for s in doc.slots:
        overlapping = [c for c in report.checked
                       if c.claim.start < s.end and c.claim.end > s.start]
        matched.update(id(c) for c in overlapping)
        rows.append({
            "family": s.family, "template": s.template, "metric": s.metric,
            "error": s.error,
            "extracted": bool(overlapping),
            "flagged": any(c.status == CONTRADICTED for c in overlapping),
        })
    # Contradictions that don't correspond to any slot are false alarms too.
    spurious = sum(1 for c in report.checked
                   if id(c) not in matched and c.status == CONTRADICTED)
    return rows, spurious


def _rate(num, den):
    return round(num / den, 4) if den else None


def summarize(rows, spurious):
    injected = [r for r in rows if r["error"]]
    clean = [r for r in rows if not r["error"]]
    tp = sum(r["flagged"] for r in injected)
    fp = sum(r["flagged"] for r in clean) + spurious

    def recall(rs):
        return {"n": len(rs), "recall": _rate(sum(r["flagged"] for r in rs), len(rs))}

    by_error, by_family, by_template = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in injected:
        by_error[r["error"]].append(r)
        by_family[r["family"]].append(r)
        by_template[r["template"]].append(r)
    clean_by_family = defaultdict(list)
    for r in clean:
        clean_by_family[r["family"]].append(r)

    return {
        "slots": len(rows),
        "injected": len(injected),
        "clean": len(clean),
        "true_positives": tp,
        "false_positives": fp,
        "spurious_contradictions": spurious,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, len(injected)),
        "false_positive_rate_clean": _rate(sum(r["flagged"] for r in clean), len(clean)),
        "recall_by_family": {k: recall(v) for k, v in sorted(by_family.items())},
        "recall_by_error": {k: recall(v) for k, v in sorted(by_error.items())},
        "recall_by_template": {k: recall(v) for k, v in sorted(by_template.items())},
        "extraction_coverage_clean": {
            k: {"n": len(v), "coverage": _rate(sum(r["extracted"] for r in v), len(v))}
            for k, v in sorted(clean_by_family.items())
        },
    }


def run(snapshot=None, seed=SEED, docs_per_ticker=DOCS_PER_TICKER):
    snapshot = snapshot or load_snapshot()
    facts = snapshot["facts"]
    tickers = sorted(facts)
    rng = random.Random(seed)
    all_rows, spurious = [], 0
    for ticker in tickers:
        for _ in range(docs_per_ticker):
            peer = rng.choice([t for t in tickers if t != ticker])
            b = Builder(rng, facts, ticker, peer)
            doc = Doc()
            doc.add(f"## {ticker} review\n\n")
            for template in rng.sample(STANDARD, 4) + [rng.choice(HARD)]:
                template(b, doc)
            rows, extra = score_doc(doc, facts, ticker)
            all_rows.extend(rows)
            spurious += extra
    summary = summarize(all_rows, spurious)
    summary["seed"] = seed
    summary["docs"] = len(tickers) * docs_per_ticker
    summary["snapshot_taken_at"] = snapshot.get("taken_at")
    return summary


def to_markdown(s):
    lines = [
        "# Injected-error evaluation of the claim verifier",
        "",
        f"{s['docs']} synthetic narratives over the facts snapshot "
        f"({s['snapshot_taken_at']}), seed {s['seed']}. "
        f"{s['injected']} injected errors among {s['slots']} numeric/verdict statements.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {s['precision']:.1%} |",
        f"| Recall (all) | {s['recall']:.1%} |",
        f"| False-positive rate on clean statements | {s['false_positive_rate_clean']:.1%} |",
        f"| Spurious contradictions (no matching statement) | {s['spurious_contradictions']} |",
        "",
        "**Recall by template family** (standard = phrasings the extractor targets; "
        "hard = phrasings it was not designed for)",
        "",
        "| Family | Injected | Recall | Clean coverage |",
        "|---|---|---|---|",
    ]
    for fam, r in s["recall_by_family"].items():
        cov = s["extraction_coverage_clean"].get(fam, {}).get("coverage")
        lines.append(f"| {fam} | {r['n']} | {r['recall']:.1%} | "
                     f"{'' if cov is None else f'{cov:.1%}'} |")
    lines += ["", "**Recall by error type**", "", "| Error | Injected | Recall |", "|---|---|---|"]
    for err, r in s["recall_by_error"].items():
        lines.append(f"| {err} | {r['n']} | {r['recall']:.1%} |")
    lines += ["", "**Recall by template**", "", "| Template | Injected | Recall |", "|---|---|---|"]
    for t, r in s["recall_by_template"].items():
        lines.append(f"| {t} | {r['n']} | {r['recall']:.1%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    summary = run()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "injection_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = to_markdown(summary)
    with open(os.path.join(RESULTS_DIR, "injection_eval.md"), "w") as f:
        f.write(md)
    print(md)
