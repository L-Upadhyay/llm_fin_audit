"""
CSP-based financial constraint checker.

Implements the AIMA-style classical CSP loop — domain pruning via unary
constraints, AC-3 arc consistency on binary constraints, and backtracking
search with forward checking — and applies it to financial-ratio
classification.

Each ratio (debt_to_equity, current_ratio, interest_coverage_ratio) is a CSP
variable with the domain {healthy, warning, critical}. The actual numeric
value of the ratio prunes the domain via unary constraints; the solver then
assigns a status to each variable, and the highest-severity status decides
the overall verdict (PASS / WARNING / FAIL).
"""

from collections import deque


# Domain values, ordered from least to most severe. Order matters: backtracking
# iterates in domain order, so unconstrained variables default to "healthy"
# rather than the search arbitrarily picking "critical".
DEFAULT_DOMAIN = ["healthy", "warning", "critical"]


class Variable:
    """A CSP variable: a name plus the list of values it can still take."""

    def __init__(self, name, domain):
        self.name = name
        self.domain = list(domain)

    def __repr__(self):
        return f"Variable({self.name}, domain={self.domain})"


class Constraint:
    """
    A constraint over one or more variables.

    `variables`   list of variable names this constraint touches.
    `rule`        callable: rule(assignment_dict) -> bool. The dict maps
                  variable names to candidate values for (at least) every
                  name in `variables`. Returns True iff the candidate values
                  jointly satisfy the constraint.
    `description` human-readable label, used for debugging.
    """

    def __init__(self, variables, rule, description=""):
        self.variables = list(variables)
        self.rule = rule
        self.description = description

    def __repr__(self):
        return f"Constraint({self.variables}: {self.description!r})"


class FinancialCSP:
    """
    A small CSP framework specialized for financial-ratio classification.

    Public surface:
        add_variable(name, domain)
        add_constraint(constraint)
        ac3()                # arc consistency on binary constraints
        backtrack()          # backtracking search + forward checking
        solve(ratios_dict)   # build CSP from ratios, run pipeline,
                             #   return PASS / WARNING / FAIL
    """

    def __init__(self):
        self.variables = {}     # name -> Variable
        self.constraints = []   # list of Constraint

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def add_variable(self, name, domain):
        """Register a variable with its initial domain."""
        self.variables[name] = Variable(name, domain)

    def add_constraint(self, constraint):
        """Register a constraint (unary, binary, or higher arity)."""
        self.constraints.append(constraint)

    # ------------------------------------------------------------------ #
    # Domain pruning
    # ------------------------------------------------------------------ #

    def _apply_unary_constraints(self):
        """
        Walk every unary constraint and delete from the variable's domain any
        value that fails the rule. AC-3 itself only handles binary arcs, so
        unary pruning is done up front.
        """
        for c in self.constraints:
            if len(c.variables) == 1:
                name = c.variables[0]
                var = self.variables[name]
                var.domain = [v for v in var.domain if c.rule({name: v})]

    def ac3(self):
        """
        AC-3 arc-consistency algorithm.

        Build a queue of directed arcs (Xi, Xj) from every binary constraint.
        For each arc, remove from Xi.domain any value lacking a compatible
        partner in Xj.domain; if Xi shrinks, re-enqueue arcs (Xk, Xi) for
        every other neighbor Xk so the propagation continues. Returns False
        if any domain is wiped out (the CSP is unsatisfiable), True otherwise.

        Unary constraints are applied first as a convenience; with no binary
        constraints in this project AC-3 is effectively a no-op, but the full
        algorithm is implemented so that cross-ratio rules can be added later
        without touching the engine.
        """
        self._apply_unary_constraints()

        binary = [c for c in self.constraints if len(c.variables) == 2]
        queue = deque()
        for c in binary:
            a, b = c.variables
            queue.append((a, b, c))
            queue.append((b, a, c))

        while queue:
            xi, xj, c = queue.popleft()
            if self._revise(xi, xj, c):
                if not self.variables[xi].domain:
                    return False
                # Xi shrank — recheck arcs pointing into Xi from its
                # other neighbors.
                for other in binary:
                    if other is c or xi not in other.variables:
                        continue
                    xk = other.variables[0] if other.variables[1] == xi else other.variables[1]
                    if xk != xj:
                        queue.append((xk, xi, other))
        return True

    def _revise(self, xi, xj, constraint):
        """
        Remove from Xi.domain every value with no supporting partner in
        Xj.domain. Returns True if the domain changed.
        """
        revised = False
        for x in list(self.variables[xi].domain):
            has_support = any(
                constraint.rule({xi: x, xj: y})
                for y in self.variables[xj].domain
            )
            if not has_support:
                self.variables[xi].domain.remove(x)
                revised = True
        return revised

    # ------------------------------------------------------------------ #
    # Backtracking search
    # ------------------------------------------------------------------ #

    def backtrack(self, assignment=None):
        """
        Backtracking search with forward checking.

        Pick the next unassigned variable, try each value in its domain in
        order, and for each consistent value run forward checking — prune
        every neighbor's domain to remove values that can't co-exist with
        the current partial assignment. Recurse; on failure restore the
        snapshotted domains and try the next value.

        Returns a complete assignment dict (var name -> value), or None if
        no consistent assignment exists.
        """
        if assignment is None:
            assignment = {}

        # Goal test: every variable assigned.
        if len(assignment) == len(self.variables):
            return dict(assignment)

        # Variable ordering: first unassigned. (MRV would be a refinement.)
        unassigned = [n for n in self.variables if n not in assignment]
        name = unassigned[0]
        var = self.variables[name]

        for value in list(var.domain):
            assignment[name] = value

            if self._consistent_with(assignment):
                # Snapshot the other variables' domains so forward checking
                # can be undone on backtrack.
                saved = {n: list(self.variables[n].domain) for n in unassigned if n != name}

                if self._forward_check(name, assignment):
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result

                # Restore pruned domains and move to the next value.
                for n, d in saved.items():
                    self.variables[n].domain = d

            del assignment[name]

        return None

    def _consistent_with(self, assignment):
        """True iff every fully-assigned constraint is satisfied."""
        for c in self.constraints:
            if all(v in assignment for v in c.variables):
                if not c.rule(assignment):
                    return False
        return True

    def _forward_check(self, just_assigned, assignment):
        """
        After assigning `just_assigned`, prune values from each neighbor's
        domain that can no longer be made consistent with the partial
        assignment. Returns False if any neighbor's domain becomes empty.
        """
        for c in self.constraints:
            if just_assigned not in c.variables:
                continue
            for var_name in c.variables:
                if var_name in assignment:
                    continue
                var = self.variables[var_name]
                var.domain = [
                    v for v in var.domain
                    if c.rule({**assignment, var_name: v})
                ]
                if not var.domain:
                    return False
        return True

    # ------------------------------------------------------------------ #
    # End-to-end entry point
    # ------------------------------------------------------------------ #

    def solve(self, ratios_dict):
        """
        Build the financial CSP from real ratio values, run AC-3 +
        backtracking, and return an overall verdict.

        Verdict rules:
            - any variable assigned "critical"  -> FAIL
            - any variable assigned "warning"   -> WARNING
            - otherwise                          -> PASS
            - unsatisfiable CSP                  -> FAIL
        """
        # Reset state so solve() is idempotent across calls.
        self.variables = {}
        self.constraints = []

        # --- D/E ---------------------------------------------------------
        de = ratios_dict.get("debt_to_equity")
        if de is not None:
            self.add_variable("debt_to_equity", list(DEFAULT_DOMAIN))
            if de > 2.0:
                # Highly leveraged firms cannot be classified as healthy.
                self.add_constraint(Constraint(
                    ["debt_to_equity"],
                    lambda a: a["debt_to_equity"] in ("warning", "critical"),
                    "debt_to_equity > 2.0 -> warning or critical",
                ))

        # --- Current ratio ----------------------------------------------
        cr = ratios_dict.get("current_ratio")
        if cr is not None:
            self.add_variable("current_ratio", list(DEFAULT_DOMAIN))
            if cr < 1.0:
                # Cannot cover short-term obligations -> critical.
                self.add_constraint(Constraint(
                    ["current_ratio"],
                    lambda a: a["current_ratio"] == "critical",
                    "current_ratio < 1.0 -> critical",
                ))

        # --- Interest coverage ------------------------------------------
        # Skip the variable entirely if the loader couldn't compute it.
        ic = ratios_dict.get("interest_coverage_ratio")
        if ic is not None:
            self.add_variable("interest_coverage_ratio", list(DEFAULT_DOMAIN))
            if ic < 1.5:
                # Operating earnings barely cover interest -> critical.
                self.add_constraint(Constraint(
                    ["interest_coverage_ratio"],
                    lambda a: a["interest_coverage_ratio"] == "critical",
                    "interest_coverage_ratio < 1.5 -> critical",
                ))

        # --- Solve -------------------------------------------------------
        if not self.ac3():
            return "FAIL"
        assignment = self.backtrack()
        if assignment is None:
            return "FAIL"

        statuses = set(assignment.values())
        if "critical" in statuses:
            return "FAIL"
        if "warning" in statuses:
            return "WARNING"
        return "PASS"
