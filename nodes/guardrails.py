"""
nodes/guardrails.py

Guardrails node — runs after synthesis, before evaluation.

Performs two passes:
1. Rule-based scan: fast keyword checks across all agent outputs
   for HIPAA violations, hallucination signals, and clinical red flags.
2. Gemini-based check: sends the synthesis text to the LLM to catch
   subtler issues the keyword scan would miss (e.g. contradictions
   between research and product claims, unrealistic timelines).

Writes to state["guardrail_flags"] — accumulated via _merge_lists reducer.
Non-fatal: the graph always continues regardless of flags found.
Critical flags trigger the route_after_guardrails edge in edges.py
to skip evaluation and go straight to report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from graph.state import CompanyState


# ── config ────────────────────────────────────────────────────────────────────

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE = 0.1   # low temp for consistent flag detection


# ── rule-based keyword checks ─────────────────────────────────────────────────

_HIPAA_RULES = [
    ("store phi",            "PHI storage mentioned without encryption context"),
    ("unencrypted",          "Unencrypted data handling detected"),
    ("plain text password",  "Plain text credentials mentioned"),
    ("public bucket",        "Public cloud storage bucket mentioned"),
    ("no audit log",         "Audit logging gap detected"),
    ("share patient data",   "Patient data sharing without consent mechanism mentioned"),
]

_CLINICAL_RULES = [
    ("100% accuracy",        "Unrealistic clinical accuracy claim (100%)"),
    ("cure",                 "Cure claim detected — not appropriate for a SaMD product"),
    ("fda approved",         "FDA approval claimed — verify this is accurate"),
    ("clinically proven",    "Unverified clinical proof claim detected"),
]

_HALLUCINATION_SIGNALS = [
    ("as of 2019",           "Potentially stale reference year detected"),
    ("source: unknown",      "Unknown source cited — possible hallucination"),
    ("studies show",         "Vague studies reference without citation"),
]


def _rule_based_scan(state: CompanyState) -> list[str]:
    """
    Concatenates all agent output text and runs keyword rules over it.
    Fast, free, deterministic.
    """
    # Gather all agent output text into one searchable blob
    blob = " ".join([
        json.dumps(state.get("market_output") or {}),
        json.dumps(state.get("research_output") or {}),
        json.dumps(state.get("product_output") or {}),
        json.dumps(state.get("architecture_output") or {}),
        state.get("synthesis") or "",
    ]).lower()

    flags = []
    for keyword, message in [*_HIPAA_RULES, *_CLINICAL_RULES, *_HALLUCINATION_SIGNALS]:
        if keyword in blob:
            flags.append(f"[guardrails:rule] {message}")

    return flags


# ── LLM-based check ───────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """You are a MedTech compliance and startup viability auditor.

You will receive a startup brief. Your job is to identify ONLY genuine red flags — 
issues that would materially harm the startup, mislead investors, or create regulatory risk.

Respond in valid JSON only. No markdown fences. No preamble.

Schema:
{
  "flags": [
    {
      "severity": "critical | warning | info",
      "category": "hipaa | clinical | regulatory | business | hallucination",
      "message": "Concise description of the issue"
    }
  ]
}

If no flags found, return: {"flags": []}

Be strict about what counts as a red flag. Do not flag normal startup risks or 
uncertainty — only concrete problems in what is written.
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=TEMPERATURE,
        convert_system_message_to_human=False,
    )


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_llm(llm: ChatGoogleGenerativeAI, messages: list) -> str:
    return llm.invoke(messages).content


def _llm_scan(synthesis: str) -> list[str]:
    """
    Sends the synthesis text to Gemini for a deep compliance/hallucination check.
    Returns formatted flag strings or an empty list on failure.
    """
    llm = _get_llm()
    messages = [
        SystemMessage(content=_LLM_SYSTEM_PROMPT),
        HumanMessage(content=f"Audit this startup brief:\n\n{synthesis}"),
    ]

    try:
        raw = _call_llm(llm, messages)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()

        parsed = json.loads(cleaned)
        flags = parsed.get("flags", [])

        return [
            f"[guardrails:llm:{f.get('severity','?')}] "
            f"[{f.get('category','?')}] {f.get('message','?')}"
            for f in flags
        ]

    except Exception as e:
        # LLM scan failure is non-fatal — rule scan already ran
        return [f"[guardrails:llm] Scan failed — {e}"]


# ── node ──────────────────────────────────────────────────────────────────────

def guardrails_node(state: CompanyState) -> dict:
    """
    LangGraph node for guardrails.
    Runs rule-based + LLM-based checks.
    Writes accumulated flags to state["guardrail_flags"].
    """
    print("[guardrails] Running compliance and hallucination checks...")

    # Pass 1: fast rule-based scan
    rule_flags = _rule_based_scan(state)
    if rule_flags:
        print(f"[guardrails] Rule scan found {len(rule_flags)} flag(s).")

    # Pass 2: LLM deep scan (only if synthesis exists)
    synthesis = state.get("synthesis") or ""
    llm_flags = []
    if synthesis:
        print("[guardrails] Running LLM compliance audit...")
        llm_flags = _llm_scan(synthesis)
        if llm_flags:
            print(f"[guardrails] LLM scan found {len(llm_flags)} flag(s).")

    all_flags = rule_flags + llm_flags

    if not all_flags:
        print("[guardrails] No flags found.")

    print("[guardrails] Done.")
    return {"guardrail_flags": all_flags}