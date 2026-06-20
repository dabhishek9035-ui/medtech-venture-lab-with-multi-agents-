"""
agents/research_agent.py

Research Agent — analyses clinical evidence, benchmarks, research gaps,
and recommended validation study design for the given MedTech startup idea.

Mirrors the structure of market_agent.py exactly so all 4 agents
are consistent and easy to maintain. The only differences are:
- prompt file (prompts/research.txt)
- output key (state["research_output"])
- agent name in log messages
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from graph.state import CompanyState


# ── config ────────────────────────────────────────────────────────────────────

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "research.txt"


# ── LLM client ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=TEMPERATURE,
        convert_system_message_to_human=False,
    )


# ── retry wrapper ─────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(1),
    reraise=True,
)
def _call_llm(llm: ChatGoogleGenerativeAI, messages: list) -> str:
    response = llm.invoke(messages)
    return response.content


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(raw: str, agent_name: str) -> tuple[dict | None, str | None]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        return None, f"{agent_name}: JSON parse error — {e}. Raw response: {cleaned[:200]}"


# ── node ──────────────────────────────────────────────────────────────────────

def research_node(state: CompanyState) -> dict:
    """
    LangGraph node for the Research Agent.
    Reads state["idea"] (enriched framing from orchestrator).
    Writes to state["research_output"].
    """
    print("[research_agent] Starting clinical research analysis...")

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    idea = state.get("idea", "")

    llm = _get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyse this startup idea:\n\n{idea}"),
    ]

    errors: list[str] = []

    try:
        raw = _call_llm(llm, messages)
        parsed, parse_error = _parse_json(raw, "research_agent")

        if parse_error:
            errors.append(parse_error)
            print(f"[research_agent] Parse error: {parse_error}")
            return {"research_output": None, "errors": errors}

        print("[research_agent] Done.")
        return {"research_output": parsed, "errors": errors}

    except Exception as e:
        error_msg = f"research_agent: LLM call failed after retries — {e}"
        print(f"[research_agent] Error: {error_msg}")
        return {"research_output": None, "errors": [error_msg]}