"""
agents/product_agent.py

Product Agent — analyses MVP feature set, user personas, UX flow,
differentiators, and success metrics for the given MedTech startup idea.

Mirrors market_agent.py / research_agent.py in structure.
Output key: state["product_output"]
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
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "product.txt"


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

def product_node(state: CompanyState) -> dict:
    """
    LangGraph node for the Product Agent.
    Reads state["idea"] (enriched framing from orchestrator).
    Writes to state["product_output"].
    """
    print("[product_agent] Starting product analysis...")

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
        parsed, parse_error = _parse_json(raw, "product_agent")

        if parse_error:
            errors.append(parse_error)
            print(f"[product_agent] Parse error: {parse_error}")
            return {"product_output": None, "errors": errors}

        print("[product_agent] Done.")
        return {"product_output": parsed, "errors": errors}

    except Exception as e:
        error_msg = f"product_agent: LLM call failed after retries — {e}"
        print(f"[product_agent] Error: {error_msg}")
        return {"product_output": None, "errors": [error_msg]}