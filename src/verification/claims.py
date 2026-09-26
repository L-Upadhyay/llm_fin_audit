"""
Claim extraction: find the checkable statements in free-form LLM text.

A *claim* is a number (or verdict label) that the text attaches to a known
metric and ticker, e.g. "Apple's current ratio is 1.43" or
"| Debt-to-Equity | 1.34 | 0.18 |". Extraction is deliberately rule-based
and deterministic: the verifier must not depend on another LLM.

Handled layouts:
  - prose        "the current ratio of 0.89 is below 1.0"
  - tables       markdown tables, with tickers in the header row
  - headings     "**Debt-to-Equity Ratio**" followed by "* Value: 0.12"

Numbers that follow words like "below", "above", "exceeds" become
*comparison* claims ("current ratio is below 1.0" is checked as
actual < 1.0). Numbers after reference words ("threshold", "industry
average", "typically") are skipped: they describe a benchmark, not the
company.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------- #
# Metric registry
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class MetricSpec:
    key: str        # key inside the per-ticker facts
    source: str     # "ratios" or "realtime"
    kind: str       # "multiple" | "fraction" | "percent" | "currency"
    aliases: tuple  # regex fragments, matched case-insensitively
    label: str


# Order matters only for readability; overlapping aliases are resolved by
# preferring the longest match (so "gross profit margin" beats "profit margin").
METRICS = (
    MetricSpec("debt_to_equity", "ratios", "multiple",
               (r"debt[\s-]*to[\s-]*equity(?:\s+ratio)?", r"\bD/E(?:\s+ratio)?\b"),
               "Debt-to-Equity"),
    MetricSpec("current_ratio", "ratios", "multiple",
               (r"\bcurrent\s+ratio",), "Current Ratio"),
    MetricSpec("quick_ratio", "ratios", "multiple",
               (r"\bquick\s+ratio", r"\bacid[\s-]*test(?:\s+ratio)?"), "Quick Ratio"),
    MetricSpec("interest_coverage_ratio", "ratios", "multiple",
               (r"\binterest\s+coverage(?:\s+ratio)?", r"\btimes\s+interest\s+earned"),
               "Interest Coverage"),
    MetricSpec("pe_ratio", "ratios", "multiple",
               (r"\bP/E(?:\s+ratio)?", r"\bPE\s+ratio",
                r"\bprice[\s-]*to[\s-]*earnings(?:\s+ratio)?"), "P/E Ratio"),
    MetricSpec("roe", "ratios", "fraction",
               (r"\breturn\s+on\s+(?:shareholders'?\s+)?equity", r"\bROE\b"),
               "Return on Equity"),
    MetricSpec("gross_margin", "ratios", "fraction",
               (r"\bgross\s+(?:profit\s+)?margins?",), "Gross Margin"),
    MetricSpec("net_profit_margin", "ratios", "fraction",
               (r"\bnet\s+(?:profit\s+|income\s+)?margins?", r"(?<!gross )\bprofit\s+margins?"),
               "Net Profit Margin"),
    MetricSpec("current_price", "realtime", "currency",
               (r"\b(?:current|share|stock|last|trading)\s+price",
                r"\btrading\s+at", r"\bpriced\s+at"), "Current Price"),
    MetricSpec("previous_close", "realtime", "currency",
               (r"\b(?:previous|prior)\s+close",), "Previous Close"),
    MetricSpec("market_cap", "realtime", "currency",
               (r"\bmarket\s+cap(?:italization)?",), "Market Cap"),
    MetricSpec("fifty_two_week_high", "realtime", "currency",
               (r"\b(?:52|fifty[\s-]two)[\s-]*week\s+high",), "52-Week High"),
    MetricSpec("fifty_two_week_low", "realtime", "currency",
               (r"\b(?:52|fifty[\s-]two)[\s-]*week\s+low",), "52-Week Low"),
    MetricSpec("dividend_yield", "realtime", "percent",
               (r"\bdividend\s+yield",), "Dividend Yield"),
    MetricSpec("beta", "realtime", "multiple",
               (r"\bbeta\b",), "Beta"),
)

METRICS_BY_KEY = {m.key: m for m in METRICS}

_ALIAS_RE = re.compile(
    "|".join(f"(?P<m{i}_{j}>{alias})"
             for i, m in enumerate(METRICS) for j, alias in enumerate(m.aliases)),
    re.IGNORECASE,
)


def _metric_for_match(match):
    name = match.lastgroup  # "m{i}_{j}"
    return METRICS[int(name[1:].split("_")[0])]


# ---------------------------------------------------------------------- #
# Numbers
# ---------------------------------------------------------------------- #

_NUMBER_RE = re.compile(
    r"(?P<sign>-|−|\bnegative\s+|\bminus\s+)?"
    r"(?:(?P<cur>\$)\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)"
    r"(?:\s?(?P<suffix>%|percent\b|x\b|×|times\b|trillion\b|billion\b|million\b|"
    r"thousand\b|bn\b|mn\b|T\b|B\b|M\b|K\b))?",
    re.IGNORECASE,
)

_SCALE = {
    "trillion": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

_COMPARATOR_RE = re.compile(
    r"\b(?P<lt>below|under|less\s+than|lower\s+than|beneath|short\s+of)\b|"
    r"\b(?P<gt>above|over|more\s+than|greater\s+than|higher\s+than|exceeds?|exceeding|in\s+excess\s+of)\b",
    re.IGNORECASE,
)

# Words that introduce a benchmark rather than the company's own value.
_REFERENCE_RE = re.compile(
    r"\b(threshold|benchmark|industry|sector|peers?|average|median|typical(?:ly)?|"
    r"generally|usually|target|healthy\s+range|rule\s+of\s+thumb|guideline|"
    r"recommended|ideal(?:ly)?|norm|standard)\b",
    re.IGNORECASE,
)


@dataclass
class Number:
    value: float          # value in natural units (percent kept as percent number)
    start: int
    end: int
    raw: str
    decimals: int         # digits after the decimal point, as written
    is_percent: bool
    scale: float          # multiplier applied for T/B/M suffixes (1.0 if none)


def parse_numbers(text, start=0, end=None):
    """All numbers in text[start:end], skipping years and durations."""
    end = len(text) if end is None else end
    out = []
    for m in _NUMBER_RE.finditer(text, start, end):
        num = m.group("num")
        after = text[m.end():m.end() + 12].lower()
        suffix = (m.group("suffix") or "").lower()
        # Durations/labels: "52-week", "8 quarters", "5 years", "Q4".
        if re.match(r"^[\s-]*(week|weeks|day|days|month|months|quarter|quarters|year|years)\b", after):
            continue
        if text[max(0, m.start() - 1):m.start()].lower() == "q":
            continue
        value = float(num.replace(",", ""))
        # Bare four-digit integers in a plausible year range are dates.
        if (not m.group("cur") and not suffix and "." not in num
                and "," not in num and 1990 <= value <= 2100):
            continue
        if m.group("sign"):
            value = -value
        scale = _SCALE.get(suffix, 1.0)
        decimals = len(num.split(".")[1]) if "." in num else 0
        out.append(Number(
            value=value * scale, start=m.start(), end=m.end(), raw=m.group(0).strip(),
            decimals=decimals, is_percent=suffix in ("%", "percent"), scale=scale,
        ))
    return out


# ---------------------------------------------------------------------- #
# Claims
# ---------------------------------------------------------------------- #

VERDICT_ALIASES = {
    "PASS": "PASS", "WARNING": "WARNING", "FAIL": "FAIL",
    "HOLD": "PASS", "WATCH": "WARNING", "AVOID": "FAIL",
    "INSUFFICIENT_DATA": "INSUFFICIENT_DATA",
}
_VERDICT_RE = re.compile(r"\b(PASS|WARNING|FAIL|HOLD|WATCH|AVOID|INSUFFICIENT_DATA)\b")


@dataclass
class Claim:
    kind: str                 # "value" | "comparison" | "verdict"
    metric: str               # metric key, or "verdict"
    ticker: str
    start: int                # span of the number / label in the text
    end: int
    raw: str
    number: Number = None     # for value / comparison claims
    comparator: str = None    # "lt" / "gt" for comparison claims
    verdict: str = None       # normalized verdict for verdict claims
    context: str = field(default="", repr=False)


def _sentence_bounds(text, pos):
    """[start, end) of the sentence or line containing `pos`."""
    boundary = re.compile(r"(?<!\d)\.(?!\d)|[!?;\n]")
    start = 0
    for m in boundary.finditer(text, 0, pos):
        start = m.end()
    m = boundary.search(text, pos)
    return start, (m.start() if m else len(text))


def _ticker_positions(text, tickers):
    """Sorted (pos, ticker) for every mention of a known ticker symbol."""
    found = []
    for t in tickers:
        for m in re.finditer(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", text):
            found.append((m.start(), t))
    return sorted(found)


def _ticker_at(pos, mentions, primary):
    """The ticker most recently mentioned before `pos`, else `primary`."""
    current = primary
    for p, t in mentions:
        if p > pos:
            break
        current = t
    return current


def _is_heading_line(line):
    s = line.strip()
    return bool(s) and len(s) <= 70 and (
        s.startswith(("#", "**", "__")) or re.match(r"^\d+[.)]\s", s) or s.endswith(":")
    )


def _table_claims(text, tickers, primary, table_spans):
    """Claims from markdown tables. Header cells naming tickers pick the column."""
    claims = []
    lines = text.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        block_start = i
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            i += 1
        block = range(block_start, i)
        table_spans.append((offsets[block_start], offsets[i - 1] + len(lines[i - 1])))

        header = [c.strip() for c in lines[block_start].strip().strip("|").split("|")]
        col_ticker = {}
        for ci, cell in enumerate(header):
            for t in tickers:
                if re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", cell):
                    col_ticker[ci] = t

        for li in block:
            line = lines[li]
            if re.match(r"^\s*\|?\s*:?-{2,}", line):
                continue
            cells, cpos = [], offsets[li]
            # Recover each cell's absolute span.
            for m in re.finditer(r"\|([^|]*)", line):
                cells.append((m.group(1), cpos + m.start(1)))
            if not cells:
                continue
            first = cells[0][0]
            alias = _ALIAS_RE.search(first)
            if not alias:
                continue
            spec = _metric_for_match(alias)
            for ci, (cell, cstart) in enumerate(cells[1:], start=1):
                if not cell.strip():
                    continue
                nums = parse_numbers(text, cstart, cstart + len(cell))
                if not nums:
                    continue
                ticker = col_ticker.get(ci, primary if len(col_ticker) == 0 else None)
                if ticker is None:
                    continue
                n = nums[0]
                claims.append(Claim("value", spec.key, ticker, n.start, n.end, n.raw,
                                    number=n, context=line.strip()))
    return claims


def extract_claims(text, tickers, primary):
    """
    Extract every checkable claim from `text`.

    `tickers` is the list of symbols the facts cover (the first mention
    before a claim decides which ticker it is about); `primary` is used
    when no ticker has been mentioned yet.
    """
    tickers = [t.upper() for t in tickers]
    primary = primary.upper()
    table_spans = []
    claims = _table_claims(text, tickers, primary, table_spans)

    def in_table(pos):
        return any(a <= pos < b for a, b in table_spans)

    mentions = _ticker_positions(text, tickers)
    alias_matches = [m for m in _ALIAS_RE.finditer(text) if not in_table(m.start())]

    for idx, m in enumerate(alias_matches):
        spec = _metric_for_match(m)
        next_alias = alias_matches[idx + 1].start() if idx + 1 < len(alias_matches) else len(text)
        # Next mention of a *different* metric: a heading's explanatory
        # sentence often repeats its own metric name before the value.
        next_other = next(
            (a.start() for a in alias_matches[idx + 1:] if _metric_for_match(a) is not spec),
            len(text),
        )
        s_start, s_end = _sentence_bounds(text, m.start())
        window_end = min(s_end, next_alias, m.end() + 90)
        nums = parse_numbers(text, m.end(), window_end)

        # Heading followed by bullets: "**Current Ratio**\n* Value: 1.43"
        if not nums:
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line_end = len(text) if line_end == -1 else line_end
            if _is_heading_line(text[line_start:line_end]):
                look_end = next_other
                # At most the next three non-empty lines.
                seen, cursor = 0, line_end
                while seen < 3 and cursor < look_end:
                    nl = text.find("\n", cursor + 1)
                    nl = look_end if nl == -1 else min(nl, look_end)
                    if text[cursor + 1:nl].strip():
                        seen += 1
                        nums = parse_numbers(text, cursor + 1, nl)
                        if nums:
                            s_start = cursor + 1
                            break
                    cursor = nl

        if not nums:
            continue
        n = nums[0]
        between = text[m.end():n.start]
        if _REFERENCE_RE.search(between):
            continue
        comp = _COMPARATOR_RE.search(between)
        ticker = _ticker_at(m.start(), mentions, primary)
        context = text[s_start:max(n.end, min(s_end, len(text)))].strip()
        if comp:
            claims.append(Claim("comparison", spec.key, ticker, n.start, n.end, n.raw,
                                number=n, comparator="lt" if comp.group("lt") else "gt",
                                context=context))
        else:
            claims.append(Claim("value", spec.key, ticker, n.start, n.end, n.raw,
                                number=n, context=context))

    # A heading and its explanatory sentence can both resolve to the same
    # number; keep one claim per span.
    seen_spans, unique = set(), []
    for c in claims:
        if (c.start, c.end) not in seen_spans:
            seen_spans.add((c.start, c.end))
            unique.append(c)
    claims = unique

    for m in _VERDICT_RE.finditer(text):
        claims.append(Claim("verdict", "verdict", _ticker_at(m.start(), mentions, primary),
                            m.start(), m.end(), m.group(0),
                            verdict=VERDICT_ALIASES[m.group(0)]))

    claims.sort(key=lambda c: c.start)
    return claims


def untracked_numbers(text, claims):
    """Numbers in `text` not attached to any claim — a coverage signal."""
    used = {(c.start, c.end) for c in claims}
    return [n for n in parse_numbers(text) if (n.start, n.end) not in used]
