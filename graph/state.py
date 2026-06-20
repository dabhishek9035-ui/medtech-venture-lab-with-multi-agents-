"""
graph/state.py

Defines CompanyState — the single shared TypedDict that flows through
every node in the LangGraph graph.

Rules:
- Every agent reads from this state and writes only its own output key.
- No agent writes to another agent's key.
- The synthesis node is the only node that reads all 4 agent output keys.
- Reducers (Annotated fields) handle parallel writes from the fan-out agents.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


def _keep_last(a: Any, b: Any) -> Any:
    """Reducer: when two parallel agents write the same key, keep the latest."""
    return b if b is not None else a


def _merge_lists(a: list, b: list) -> list:
    """Reducer: merge two lists without duplicates, preserving order."""
    seen = set()
    result = []
    for item in (a or []) + (b or []):
        key = str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


class CompanyState(TypedDict):
    # ── input ────────────────────────────────────────────────────────────────
    idea: str
    """The raw startup idea provided by the user."""

    domain: str
    """Target domain, e.g. 'Healthcare / MedTech'."""

    # ── agent outputs (written in parallel, reducers handle merge) ────────────
    market_output: Annotated[Optional[dict], _keep_last]
    """
    Output from the Market Agent.
    Keys: tam, competitors, regulatory_landscape, market_risks
    """

    research_output: Annotated[Optional[dict], _keep_last]
    """
    Output from the Research Agent.
    Keys: clinical_evidence, benchmarks, research_gaps, key_papers
    """

    product_output: Annotated[Optional[dict], _keep_last]
    """
    Output from the Product Agent.
    Keys: mvp_features, user_personas, ux_flow, differentiators
    """

    architecture_output: Annotated[Optional[dict], _keep_last]
    """
    Output from the Architecture Agent.
    Keys: tech_stack, data_pipeline, hipaa_considerations, infra_diagram_description
    """

    # ── pipeline outputs (written sequentially) ───────────────────────────────
    synthesis: Optional[str]
    """
    Merged narrative produced by the Synthesis node.
    Plain text combining all 4 agent outputs into a coherent startup brief.
    """

    guardrail_flags: Annotated[list[str], _merge_lists]
    """
    List of issues raised by the Guardrails node.
    e.g. ['PHI stored without encryption mentioned', 'Unverified clinical claim detected']
    """

    viability_score: Optional[int]
    """
    Integer 0–100 produced by the Evaluator node.
    Higher = stronger startup viability signal.
    """

    final_report: Optional[str]
    """
    Fully rendered Markdown report produced by the Report node.
    This is what gets written to disk by main.py.
    """

    # ── error tracking ────────────────────────────────────────────────────────
    errors: Annotated[list[str], _merge_lists]
    """
    Accumulated errors from any node (agent failures, parse errors, etc.).
    Non-fatal — the graph continues even if one agent errors out.
    """