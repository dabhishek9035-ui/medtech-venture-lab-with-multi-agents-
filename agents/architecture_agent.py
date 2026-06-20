"""
agents/architecture_agent.py

Architecture Agent — analyses tech stack, data pipeline, HIPAA compliance
requirements, scalability, and integration points for the given MedTech startup.

Mirrors market_agent.py / research_agent.py in structure.
Output key: state["architecture_output"]

HIPAA note:
    This agent is the primary source of compliance-related output.
    The guardrails node (nodes/guardrails.py) scans this output specifically
    for red flags like unencrypted PHI storage or missing audit logging.
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
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "architecture.txt"


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


# ── HIPAA red-flag pre-scan ───────────────────────────────────────────────────

_HIPAA_RED_FLAGS = [
    ("unencrypted", "PHI stored or transmitted without encryption mentioned"),
    ("no audit",    "Audit logging not addressed in architecture"),
    ("public s3",   "S3 bucket may be publicly accessible"),
    ("plain text",  "Credentials or PHI stored in plain text"),
]

def _scan_hipaa_flags(output: dict) -> list[str]:
    """
    Quick keyword scan of the architecture output for obvious HIPAA red flags.
    Returns a list of flag strings to be merged into state["guardrail_flags"].
    These are surfaced to the user in the final report and terminal output.
    """
    raw_text = json.dumps(output).lower()
    flags = []
    for keyword, message in _HIPAA_RED_FLAGS:
        if keyword in raw_text:
            flags.append(f"[architecture] Possible issue: {message}")
    return flags


# ── node ──────────────────────────────────────────────────────────────────────

def architecture_node(state: CompanyState) -> dict:
    """
    LangGraph node for the Architecture Agent.
    Reads state["idea"] (enriched framing from orchestrator).
    Writes to state["architecture_output"] and may append to state["guardrail_flags"].
    """
    print("[architecture_agent] Starting technical architecture analysis...")

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
        parsed, parse_error = _parse_json(raw, "architecture_agent")

        if parse_error:
            errors.append(parse_error)
            print(f"[architecture_agent] Parse error: {parse_error}")
            return {"architecture_output": None, "errors": errors, "guardrail_flags": []}

        # Pre-scan for HIPAA red flags before synthesis node sees the output
        hipaa_flags = _scan_hipaa_flags(parsed)
        if hipaa_flags:
            print(f"[architecture_agent] HIPAA flags detected: {hipaa_flags}")

        print("[architecture_agent] Done.")
        return {
            "architecture_output": parsed,
            "errors": errors,
            "guardrail_flags": hipaa_flags,
        }

    except Exception as e:
        error_msg = f"architecture_agent: LLM call failed after retries — {e}"
        print(f"[architecture_agent] Error: {error_msg}")
        return {
            "architecture_output": None,
            "errors": [error_msg],
            "guardrail_flags": [],
        }