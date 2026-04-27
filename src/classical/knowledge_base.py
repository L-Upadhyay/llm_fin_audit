"""
Forward-chaining knowledge base for financial compliance rules.

Implements a small Horn-clause inference engine: each Clause is
"IF body_1 AND body_2 ... THEN head". `tell()` asserts a base fact,
`ask()` checks whether a fact is known or already derived, and
`forward_chain()` applies clauses repeatedly until reaching a fixed point.

Used by the compliance layer to flag firms whose ratio profile triggers
a chain of risk rules ending in `flag_for_review`.
"""


class Clause:
    """
    A Horn clause: IF every fact in `body` is known, THEN `head` follows.

    `head` is a fact name (string).
    `body` is a list of fact names — empty body means the head is itself a fact.
    """

    def __init__(self, head, body):
        self.head = head
        self.body = list(body)

    def __repr__(self):
        body_str = " AND ".join(self.body) if self.body else "TRUE"
        return f"IF {body_str} THEN {self.head}"


class KnowledgeBase:
    """
    A propositional KB with forward-chaining inference.

    Public surface:
        add_clause(clause)   register a rule
        tell(fact)           assert a base fact
        ask(query)           True if `query` is known or has been derived
        forward_chain()      derive everything possible; returns the list of
                             newly derived facts in derivation order
    """

    def __init__(self):
        self.clauses = []     # registered Horn clauses
        self.facts = set()    # everything known: told facts + derived facts

    def add_clause(self, clause):
        """Register a rule with the KB."""
        self.clauses.append(clause)

    def tell(self, fact):
        """Assert a base fact as known."""
        self.facts.add(fact)

    def ask(self, query):
        """Return True iff `query` is currently a known or derived fact."""
        return query in self.facts

    def forward_chain(self):
        """
        Naive forward chaining: repeatedly sweep the clause list, firing any
        clause whose body is fully entailed and whose head isn't already
        known. Stop when a full pass derives nothing new (fixed point).

        Returns the list of facts derived during this call, in derivation order.
        """
        derived = []
        changed = True
        while changed:
            changed = False
            for clause in self.clauses:
                # Skip if we already know this conclusion.
                if clause.head in self.facts:
                    continue
                # Fire iff every body fact is currently entailed.
                if all(b in self.facts for b in clause.body):
                    self.facts.add(clause.head)
                    derived.append(clause.head)
                    changed = True
        return derived


# ---------------------------------------------------------------------- #
# Pre-loaded financial compliance rules
# ---------------------------------------------------------------------- #

# Compliance rules expressed as Horn clauses.
# Two layers: ratio symptoms -> per-axis risks -> aggregate risk -> review flag.
COMPLIANCE_RULES = [
    Clause("leverage_risk",      ["debt_to_equity_high"]),
    Clause("liquidity_risk",     ["current_ratio_low"]),
    Clause("solvency_risk",      ["interest_coverage_low"]),
    Clause("high_risk_company",  ["leverage_risk", "liquidity_risk"]),
    Clause("high_risk_company",  ["solvency_risk"]),
    Clause("flag_for_review",    ["high_risk_company"]),
]


def default_compliance_kb():
    """Return a fresh KB pre-loaded with the compliance rules above."""
    kb = KnowledgeBase()
    for clause in COMPLIANCE_RULES:
        kb.add_clause(clause)
    return kb


# ---------------------------------------------------------------------- #
# End-to-end compliance check
# ---------------------------------------------------------------------- #

def run_compliance_check(ratios_dict):
    """
    Translate raw financial ratios into base facts, run forward chaining,
    and return the rules that fired plus an overall verdict.

    Verdict rules:
        - flag_for_review derived            -> FAIL
        - any rule fired (but no flag)       -> WARNING
        - no rules fired                     -> PASS

    Returns a dict:
        {
          "base_facts":       facts asserted from the ratio thresholds,
          "derived_facts":    facts produced by forward chaining (in order),
          "triggered_rules":  human-readable strings of the clauses that fired,
          "verdict":          "PASS" / "WARNING" / "FAIL",
        }
    """
    kb = default_compliance_kb()

    # --- Convert numeric ratios into base symptom facts -------------------
    base_facts = []

    de = ratios_dict.get("debt_to_equity")
    if de is not None and de > 2.0:
        # Aggressive leverage relative to equity cushion.
        kb.tell("debt_to_equity_high")
        base_facts.append("debt_to_equity_high")

    cr = ratios_dict.get("current_ratio")
    if cr is not None and cr < 1.0:
        # Short-term obligations exceed short-term assets.
        kb.tell("current_ratio_low")
        base_facts.append("current_ratio_low")

    ic = ratios_dict.get("interest_coverage_ratio")
    if ic is not None and ic < 1.5:
        # Operating earnings barely cover interest expense.
        kb.tell("interest_coverage_low")
        base_facts.append("interest_coverage_low")

    # --- Run inference ----------------------------------------------------
    derived_facts = kb.forward_chain()

    # Map each derived fact back to the clause that produced it (first match).
    triggered_rules = []
    for fact in derived_facts:
        for clause in kb.clauses:
            if clause.head == fact and all(b in kb.facts for b in clause.body):
                triggered_rules.append(str(clause))
                break

    # --- Verdict ----------------------------------------------------------
    if kb.ask("flag_for_review"):
        verdict = "FAIL"
    elif derived_facts:
        verdict = "WARNING"
    else:
        verdict = "PASS"

    return {
        "base_facts": base_facts,
        "derived_facts": derived_facts,
        "triggered_rules": triggered_rules,
        "verdict": verdict,
    }
