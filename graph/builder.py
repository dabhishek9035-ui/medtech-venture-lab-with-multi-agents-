"""
graph/builder.py

Builds and compiles the LangGraph StateGraph for the AI Company Builder.

Rate-limit note:
    Gemini free tier = 2 RPM (requests per minute) on gemini-2.0-flash.
    Running 4 agents in true parallel instantly exhausts the quota.
    PARALLEL_AGENTS env var controls behaviour:
        "true"  → parallel fan-out via Send (use when quota allows)
        "false" → sequential execution with inter-agent delay (default, safe for free tier)
"""

import os
from langgraph.graph import StateGraph, END
from langgraph.types import Send                    # fixed: was langgraph.constants

from graph.state import CompanyState
from agents.orchestrator import orchestrator_node
from agents.market_agent import market_node
from agents.research_agent import research_node
from agents.product_agent import product_node
from agents.architecture_agent import architecture_node
from nodes.synthesis import synthesis_node
from nodes.guardrails import guardrails_node
from nodes.evaluator import evaluator_node
from outputs.report import report_node

PARALLEL = os.getenv("PARALLEL_AGENTS", "false").lower() == "true"


# ── parallel mode (fan-out via Send) ──────────────────────────────────────────

def route_to_agents(state: CompanyState) -> list[Send]:
    return [
        Send("market_agent",        state),
        Send("research_agent",      state),
        Send("product_agent",       state),
        Send("architecture_agent",  state),
    ]


# ── sequential wrapper nodes (rate-limit safe) ────────────────────────────────

import time

_DELAY = float(os.getenv("AGENT_DELAY_SECONDS", "35"))

def _sequential_agents_node(state: CompanyState) -> dict:
    """
    Runs all 4 agents one after another with a configurable delay between calls.
    Default delay = 35s — safely under the 2 RPM free tier limit.
    Set AGENT_DELAY_SECONDS=0 if you have a paid quota.
    """
    updates: dict = {
        "market_output": None,
        "research_output": None,
        "product_output": None,
        "architecture_output": None,
        "errors": [],
        "guardrail_flags": [],
    }

    agents = [
        ("market_agent",        market_node,        "market_output"),
        ("research_agent",      research_node,      "research_output"),
        ("product_agent",       product_node,       "product_output"),
        ("architecture_agent",  architecture_node,  "architecture_output"),
    ]

    for i, (name, node_fn, _) in enumerate(agents):
        if i > 0:
            print(f"[sequential] Waiting {_DELAY}s before next agent (rate limit)...")
            time.sleep(_DELAY)
        result = node_fn(state)
        # merge result into updates
        for k, v in result.items():
            if k in ("errors", "guardrail_flags"):
                updates[k] = updates.get(k, []) + (v or [])
            else:
                updates[k] = v

    return updates


# ── graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(CompanyState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("synthesis",    synthesis_node)
    graph.add_node("guardrails",   guardrails_node)
    graph.add_node("evaluator",    evaluator_node)
    graph.add_node("report",       report_node)

    if PARALLEL:
        print("[builder] Mode: PARALLEL (Send fan-out)")
        graph.add_node("market_agent",       market_node)
        graph.add_node("research_agent",     research_node)
        graph.add_node("product_agent",      product_node)
        graph.add_node("architecture_agent", architecture_node)

        graph.set_entry_point("orchestrator")
        graph.add_conditional_edges(
            "orchestrator",
            route_to_agents,
            {
                "market_agent":       "market_agent",
                "research_agent":     "research_agent",
                "product_agent":      "product_agent",
                "architecture_agent": "architecture_agent",
            },
        )
        for agent in ["market_agent", "research_agent", "product_agent", "architecture_agent"]:
            graph.add_edge(agent, "synthesis")
    else:
        print("[builder] Mode: SEQUENTIAL (rate-limit safe, 35s delay between agents)")
        graph.add_node("agents", _sequential_agents_node)
        graph.set_entry_point("orchestrator")
        graph.add_edge("orchestrator", "agents")
        graph.add_edge("agents", "synthesis")

    graph.add_edge("synthesis",  "guardrails")
    graph.add_edge("guardrails", "evaluator")
    graph.add_edge("evaluator",  "report")
    graph.add_edge("report",     END)

    return graph.compile()