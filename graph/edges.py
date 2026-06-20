"""
graph/edges.py

Conditional edge functions and routing helpers for the LangGraph graph.

builder.py handles the graph wiring; this module owns the *logic*
that decides which node to go to next when the path is conditional.

Currently defines:
- route_after_guardrails: decides whether to proceed to the evaluator
  or short-circuit to the report with a warning when critical flags are found.
"""

from __future__ import annotations

from graph.state import CompanyState


# ── severity classification ───────────────────────────────────────────────────

CRITICAL_FLAG_KEYWORDS = [
    "phi without encryption",
    "unverified clinical claim",
    "illegal",
    "fda clearance misrepresented",
    "patient data exposed",
]


def _is_critical(flag: str) -> bool:
    """Returns True if a guardrail flag is classified as critical."""
    flag_lower = flag.lower()
    return any(kw in flag_lower for kw in CRITICAL_FLAG_KEYWORDS)


# ── edge functions ────────────────────────────────────────────────────────────

def route_after_guardrails(state: CompanyState) -> str:
    """
    Called as a conditional edge after the guardrails node.

    - If any CRITICAL flag exists → skip evaluation and go straight to
      'report' so the user sees the warning immediately without a viability
      score that would be misleading.
    - Otherwise → proceed to 'evaluator' as normal.

    Returns the name of the next node as a string.
    """
    flags = state.get("guardrail_flags", [])
    critical = [f for f in flags if _is_critical(f)]

    if critical:
        return "report"

    return "evaluator"


def route_after_evaluator(state: CompanyState) -> str:
    """
    Placeholder for future logic after evaluation.
    Currently always routes to 'report'.

    Could be extended to re-run specific agents if the viability
    score falls below a threshold.
    """
    score = state.get("viability_score")

    if score is not None and score < 20:
        # Very low score — still go to report but this hook lets us
        # add agent re-runs or a 'refine' node later.
        return "report"

    return "report"