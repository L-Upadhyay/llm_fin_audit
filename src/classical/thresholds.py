"""
Single source of truth for ratio thresholds and verdict labels.

The CSP solver, the compliance KB, the comparator, and the UIs all read
from here, so a threshold change in one place can't silently drift out of
sync with the others.

Each ratio has a direction and two cut-offs:
    direction "lower"  — lower is healthier. value > warning  -> warning,
                                             value > critical -> critical
    direction "higher" — higher is healthier. value < warning  -> warning,
                                              value < critical -> critical
"""

# Per-ratio status (CSP variable domain values).
HEALTHY = "healthy"
WARNING = "warning"
CRITICAL = "critical"

# Overall verdicts returned by the CSP and the KB.
PASS = "PASS"
WARN = "WARNING"
FAIL = "FAIL"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Ordering used when combining or ranking verdicts. Missing data is ranked
# with FAIL on purpose: an audit tool must never treat "unknown" as "safe".
VERDICT_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2, INSUFFICIENT_DATA: 2}

RATIO_THRESHOLDS = {
    "debt_to_equity":          {"direction": "lower",  "warning": 1.0,  "critical": 2.0},
    "current_ratio":           {"direction": "higher", "warning": 1.5,  "critical": 1.0},
    "interest_coverage_ratio": {"direction": "higher", "warning": 3.0,  "critical": 1.5},
    "quick_ratio":             {"direction": "higher", "warning": 1.0,  "critical": 0.5},
    "pe_ratio":                {"direction": "lower",  "warning": 50.0, "critical": 100.0},
    "roe":                     {"direction": "higher", "warning": 0.05, "critical": 0.0},
    "gross_margin":            {"direction": "higher", "warning": 0.20, "critical": 0.0},
    "net_profit_margin":       {"direction": "higher", "warning": 0.05, "critical": 0.0},
}

# Ratios without which no verdict is issued. Leverage and liquidity are the
# minimum balance-sheet picture; if yfinance can't supply them (bad ticker,
# rate limit, or a bank whose statements don't have current assets) the
# answer is INSUFFICIENT_DATA rather than a PASS built on nothing.
REQUIRED_RATIOS = ("debt_to_equity", "current_ratio")


def classify(metric, value):
    """Return HEALTHY / WARNING / CRITICAL for one ratio value, or None if missing."""
    if value is None:
        return None
    th = RATIO_THRESHOLDS[metric]
    if th["direction"] == "lower":
        if value > th["critical"]:
            return CRITICAL
        if value > th["warning"]:
            return WARNING
        return HEALTHY
    if value < th["critical"]:
        return CRITICAL
    if value < th["warning"]:
        return WARNING
    return HEALTHY


def missing_required(ratios_dict):
    """Names of REQUIRED_RATIOS that are absent/None in `ratios_dict`."""
    return [m for m in REQUIRED_RATIOS if ratios_dict.get(m) is None]
