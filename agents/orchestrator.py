"""
agents/orchestrator.py

The orchestrator is the first node the graph hits after user input.
It does NOT call Gemini — it is a lightweight Python node that:

1. Validates the idea is non-empty and meaningful.
2. Normalises the domain string.
3. Enriches the state with a structured problem framing that all 4
   downstream agents will use as shared context.
4. Initialises error/flag lists so reducers never receive None.

Keeping orchestration logic in Python (not LLM) means:
- Zero latency added to the critical path.
- No token cost for routing decisions.
- Deterministic behaviour — no prompt drift.

If you later need LLM-based routing (e.g. auto-detecting the domain
from the idea text), wire a Gemini call in here.
"""

from __future__ import annotations

import re
from graph.state import CompanyState


# ── constants ─────────────────────────────────────────────────────────────────

MIN_IDEA_LENGTH = 10  # characters
MAX_IDEA_LENGTH = 2000

SUPPORTED_DOMAINS = [
    "Healthcare / MedTech",
    "HealthTech",
    "BioTech",
    "Digital Health",
    "MedTech",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise_idea(idea: str) -> str:
    """Strip extra whitespace and trailing punctuation noise."""
    idea = idea.strip()
    idea = re.sub(r"\s+", " ", idea)
    return idea


def _build_problem_framing(idea: str, domain: str) -> str:
    """
    Produces a short structured framing string injected into state.
    This gives all downstream agents a consistent shared context
    without each agent re-interpreting the raw idea differently.
    """
    return (
        f"Domain: {domain}\n"
        f"Startup idea: {idea}\n\n"
        f"Analyse this idea through the lens of a {domain} startup. "
        f"Focus on clinical relevance, regulatory constraints (FDA, CE Mark, HIPAA), "
        f"and evidence-based product development."
    )


# ── node ──────────────────────────────────────────────────────────────────────

def orchestrator_node(state: CompanyState) -> dict:
    """
    LangGraph node function.
    Receives the full CompanyState, returns a partial dict of updates.
    """
    errors: list[str] = []
    idea = state.get("idea", "").strip()
    domain = state.get("domain", "Healthcare / MedTech").strip()

    # --- validate idea ---
    if not idea:
        errors.append("Orchestrator: idea is empty.")
        return {"errors": errors, "guardrail_flags": []}

    if len(idea) < MIN_IDEA_LENGTH:
        errors.append(
            f"Orchestrator: idea too short ({len(idea)} chars). "
            f"Minimum is {MIN_IDEA_LENGTH}."
        )

    if len(idea) > MAX_IDEA_LENGTH:
        idea = idea[:MAX_IDEA_LENGTH]
        errors.append(
            f"Orchestrator: idea truncated to {MAX_IDEA_LENGTH} characters."
        )

    # --- normalise ---
    idea = _normalise_idea(idea)

    if domain not in SUPPORTED_DOMAINS:
        # soft-warn, don't block
        errors.append(
            f"Orchestrator: domain '{domain}' is not in the supported list. "
            f"Proceeding anyway."
        )

    # --- build shared framing ---
    framing = _build_problem_framing(idea, domain)

    print(f"[orchestrator] Idea validated. Framing built ({len(framing)} chars).")
    print(f"[orchestrator] Routing to 4 agents in parallel...\n")

    return {
        "idea": idea,
        "domain": domain,
        # Inject framing back into idea so agents receive enriched context.
        # Agents will read state["idea"] — this replaces the raw string
        # with the structured framing.
        "idea": framing,
        "errors": errors,
        "guardrail_flags": [],
    }