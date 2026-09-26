"""
Question routing helpers: does a question need live prices, and does it
name a second ticker for comparison mode?

Kept free of agno imports so the verified pipeline and the legacy agent
team can share them.
"""

import re


# Keywords that mark a question as needing live market data. Used both for
# upstream prompt injection (so the LLM always has a fresh quote to quote
# from) and for the routing rules in the team coordinator.
_PRICE_KEYWORDS = (
    "price", "current price", "stock price", "share price",
    "today", "today's", "open", "high", "low",
    "volume",
    "52-week", "52 week", "fifty-two week", "fifty two week",
    "market cap", "market capitalization", "marketcap",
    "value", "valued", "worth", "trading at", "how much",
    "quote",
)


def is_price_question(question: str) -> bool:
    """True if the user's question is about live market data."""
    if not question:
        return False
    low = question.lower()
    return any(kw in low for kw in _PRICE_KEYWORDS)


# Common 2-5-letter uppercase words that look like tickers but aren't.
# Used by the comparison-mode detector to filter out conjunctions, pronouns,
# acronyms, and finance jargon before picking the second ticker.
_TICKER_STOPWORDS = {
    # English connectives, prepositions, pronouns, adverbs
    "OR", "AND", "VS", "FOR", "TO", "AT", "IN", "ON", "OF", "BY", "AS",
    "IF", "IS", "IT", "BE", "DO", "GO", "AM", "AN", "WE", "US", "MY",
    "ME", "HE", "SO", "NO", "OK", "THE", "ABOUT", "AROUND", "AGAIN",
    "ALSO", "EVEN", "ELSE", "OVER", "UNDER", "AFTER", "BEFORE", "FROM",
    "INTO", "ONTO", "WITH", "WITHIN", "ALONG", "ACROSS", "JUST", "ONLY",
    "VERY", "MUCH", "EVERY", "OTHER", "SAME", "THAN", "THEN", "HERE",
    "THERE", "NOW", "TODAY", "WEEK", "MONTH", "YEAR", "STOCK", "SHARE",
    "PRICE", "VALUE", "WORTH", "RATIO",
    # Common pronouns / determiners that hit the regex after upper()-ing
    # the question (e.g. "Is this company..." -> "IS THIS COMPANY...").
    "THIS", "THAT", "THESE", "THOSE", "ITS", "HIS", "HER", "HIM",
    "OUR", "OUT", "OFF", "DUE", "OWN", "ALL", "ANY", "ONE", "TWO",
    "TEN", "SIX", "FEW", "TOO", "TIE",
    # Auxiliary / state verbs
    "GET", "GOT", "PUT", "HAS", "HAD", "WAY", "USE", "SEE", "LET",
    "MAY", "OWN", "RUN", "TRY", "WHY", "YET",
    # Question words
    "WHAT", "HOW", "WHY", "WHO", "WHEN", "WHERE", "WHICH",
    # Comparison / trading verbs
    "BUY", "SELL", "HOLD", "OWN", "ADD", "DROP", "GAIN", "LOSS",
    "BETTER", "WORSE", "BEST", "WORST", "MORE", "LESS", "GOOD",
    # Common finance acronyms / unit labels
    "USD", "EUR", "GBP", "JPY", "LLC", "INC", "LTD", "CEO", "CFO", "CTO",
    "API", "ETF", "IPO", "USA", "ESG", "AI", "ML", "NLP", "PE", "EPS",
    "ROE", "ROI", "ROA", "EBIT", "FY", "YOY", "QOQ", "MOM", "DOD",
    # Auxiliary / modal verbs
    "ARE", "WAS", "HAS", "HAD", "HAVE", "WERE", "BEEN", "WILL", "WOULD",
    "COULD", "SHALL", "MIGHT", "MAY", "CAN", "SAY", "SAID",
    # Affirmation / negation
    "YES", "NOT", "ANY", "ALL", "SOME", "EACH", "BOTH",
    # Single letters that are words, not tickers
    "I", "A",
}


# A ticker is either a $cashtag in any case ("$nvda") or a 1-5 letter token
# the user actually typed in upper case ("NVDA", "F"). Matching against the
# original text — not question.upper() — is what stops ordinary words like
# "debt" or "peers" from being read as tickers.
_TICKER_RE = re.compile(r"\$([A-Za-z]{1,5})\b|\b([A-Z]{1,5})\b")

# Comparison signal — only one of these in the question puts us in compare
# mode. This prevents "Tell me about AAPL" from being misread as a
# comparison just because "TELL" or "ABOUT" matches the ticker regex.
_COMPARISON_SIGNALS = re.compile(
    r"\b(vs|versus|compare[d]?|compared\s+to|compared\s+with|comparison|"
    r"between|which\s+is|or|and|than|better|worse|stronger|weaker|"
    r"outperform[s]?|outperforming|differ[s]?|different)\b",
    re.IGNORECASE,
)


def detect_second_ticker(question: str, primary: str):
    """
    Scan the question for a second ticker symbol distinct from `primary`.

    Two-step detection: first the question must contain an explicit
    comparison signal ('vs', 'or', 'and', 'compare', 'between', etc.).
    Only then do we look for a $cashtag or an upper-case 1-5 letter token
    (as typed by the user) that isn't the primary ticker and isn't a known
    English/finance stopword. Returns None if either step fails.

    Lower-case tickers without a '$' ("aapl vs msft") are not detected —
    the trade-off for not mistaking ordinary words for symbols.
    """
    if not question:
        return None
    if not _COMPARISON_SIGNALS.search(question):
        return None
    primary = (primary or "").upper()
    for cashtag, bare in _TICKER_RE.findall(question):
        match = (cashtag or bare).upper()
        if match == primary:
            continue
        if bare and match in _TICKER_STOPWORDS:
            continue
        return match
    return None
