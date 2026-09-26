"""
Accounting-consistency checks: binary constraints between ratios that must
hold if the underlying numbers come from one consistent set of statements.

These don't judge the company — they judge the *data*. A violation usually
means the ratios were pulled from different periods (yfinance mixes the
latest annual balance sheet with trailing-twelve-month figures) or that a
source value is wrong. Either way, a downstream verdict built on them
deserves a caveat.

Each check is a `Constraint` over two or three ratio names, reusing the CSP
data structure; `check_consistency` evaluates them against the actual
values and reports the ones that fail.
"""

from src.classical.csp_solver import Constraint


# Relative slack so rounding in the source data doesn't trip a check.
_TOL = 0.01


def _le(a, b):
    return a <= b + abs(b) * _TOL + 1e-9


CONSISTENCY_CONSTRAINTS = (
    Constraint(
        ["quick_ratio", "current_ratio"],
        lambda v: _le(v["quick_ratio"], v["current_ratio"]),
        "Quick ratio exceeds current ratio. Quick assets are a subset of "
        "current assets, so the two figures likely come from different periods.",
    ),
    Constraint(
        ["net_profit_margin", "gross_margin"],
        lambda v: _le(v["net_profit_margin"], v["gross_margin"]),
        "Net margin exceeds gross margin. Only possible with large "
        "non-operating gains; check for one-off items or mismatched periods.",
    ),
    Constraint(
        ["roe", "net_profit_margin", "debt_to_equity"],
        # With positive equity (D/E > 0), ROE and net margin share a sign:
        # both are net income divided by a positive number.
        lambda v: v["debt_to_equity"] <= 0 or v["roe"] * v["net_profit_margin"] >= 0,
        "ROE and net margin have opposite signs despite positive equity; "
        "they likely cover different periods.",
    ),
    Constraint(
        ["pe_ratio", "net_profit_margin"],
        # Trailing P/E is undefined for a loss-making company.
        lambda v: v["net_profit_margin"] >= 0,
        "A trailing P/E is reported for a company with a net loss; the P/E "
        "and the margin likely cover different periods.",
    ),
)


def check_consistency(ratios):
    """
    Evaluate every consistency constraint whose ratios are all present.

    Returns a list of {"ratios": [...], "issue": str, "values": {...}} for
    each violated constraint (empty list = consistent).
    """
    issues = []
    for c in CONSISTENCY_CONSTRAINTS:
        values = {name: ratios.get(name) for name in c.variables}
        if any(v is None for v in values.values()):
            continue
        if not c.rule(values):
            issues.append({
                "ratios": list(c.variables),
                "issue": c.description,
                "values": values,
            })
    return issues
