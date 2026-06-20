"""
nodes/evaluator.py

Evaluator node — scores startup viability 0-100 using Gemini.

Runs after guardrails. Receives the full synthesis brief plus any
guardrail flags and produces:
- A numeric viability score (0-100)
- A breakdown across 5 dimensions (25 points max each, total 100)
- A one-paragraph investment summary

Score is written to state["viability_score"].
Breakdown and summary are appended to state["synthesis"] for the report.

Scoring dimensions:
  1. Market opportunity    (0-20): TAM size, timing, competition
  2. Clinical evidence     (0-20): strength of evidence, benchmark clarity
  3. Product clarity       (0-20): MVP definition, persona fit, UX logic
  4. Technical feasibility (0-20): stack soundness, HIPAA readiness, cost
  5. Risk-adjusted outlook (0-20): guardrail flags, biggest risks, mitigations
"""

from __future__ import annotations

import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from graph.state import CompanyState


# ── config ────────────────────────────────────────────────────────────────────

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE = 0.2


# ── prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a seasoned MedTech venture capital analyst scoring early-stage startup ideas.

You will receive a startup brief and a list of compliance/risk flags.
Score the startup across 5 dimensions and produce an investment summary.

Respond in valid JSON only. No markdown fences. No preamble.

Schema:
{
  "scores": {
    "market_opportunity":    { "score": 0-20, "rationale": "one sentence" },
    "clinical_evidence":     { "score": 0-20, "rationale": "one sentence" },
    "product_clarity":       { "score": 0-20, "rationale": "one sentence" },
    "technical_feasibility": { "score": 0-20, "rationale": "one sentence" },
    "risk_adjusted_outlook": { "score": 0-20, "rationale": "one sentence" }
  },
  "total": 0-100,
  "verdict": "strong | promising | needs-work | not-viable",
  "investment_summary": "2-3 sentence paragraph summarising the opportunity and key risks"
}

Be honest and critical. A score above 75 should mean genuinely investable.
Penalise heavily for unresolved HIPAA issues or weak clinical evidence.
"""


# ── LLM client ────────────────────────────────────────────────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_eval_prompt(synthesis: str, flags: list[str]) -> str:
    flags_text = (
        "\n".join(f"- {f}" for f in flags)
        if flags
        else "No flags raised."
    )
    return (
        f"## Startup Brief\n\n{synthesis}\n\n"
        f"## Compliance & Risk Flags\n\n{flags_text}\n\n"
        f"Score this startup."
    )


def _format_scorecard(parsed: dict) -> str:
    """Renders the score breakdown as a Markdown section for the report."""
    scores = parsed.get("scores", {})
    total = parsed.get("total", "?")
    verdict = parsed.get("verdict", "?").upper()
    summary = parsed.get("investment_summary", "Not available.")

    lines = [
        "\n---\n",
        "## Viability Scorecard\n",
        f"**Overall: {total}/100 — {verdict}**\n",
        "| Dimension | Score | Rationale |",
        "|-----------|-------|-----------|",
    ]

    dim_labels = {
        "market_opportunity":    "Market Opportunity",
        "clinical_evidence":     "Clinical Evidence",
        "product_clarity":       "Product Clarity",
        "technical_feasibility": "Technical Feasibility",
        "risk_adjusted_outlook": "Risk-Adjusted Outlook",
    }

    for key, label in dim_labels.items():
        dim = scores.get(key, {})
        score = dim.get("score", "?")
        rationale = dim.get("rationale", "—")
        lines.append(f"| {label} | {score}/20 | {rationale} |")

    lines += [
        "\n**Investment Summary**\n",
        summary,
    ]

    return "\n".join(lines)


# ── node ──────────────────────────────────────────────────────────────────────

def evaluator_node(state: CompanyState) -> dict:
    """
    LangGraph node for the evaluator.
    Reads state["synthesis"] and state["guardrail_flags"].
    Writes state["viability_score"] and appends scorecard to state["synthesis"].
    """
    print("[evaluator] Scoring startup viability...")

    synthesis = state.get("synthesis") or ""
    flags = state.get("guardrail_flags") or []

    if not synthesis:
        print("[evaluator] No synthesis found — skipping evaluation.")
        return {"viability_score": 0}

    llm = _get_llm()
    prompt = _build_eval_prompt(synthesis, flags)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        raw = _call_llm(llm, messages)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()

        parsed = json.loads(cleaned)
        total = int(parsed.get("total", 0))
        scorecard_md = _format_scorecard(parsed)

        # Append scorecard to synthesis so report.py gets the full picture
        updated_synthesis = synthesis + scorecard_md

        print(f"[evaluator] Score: {total}/100 — {parsed.get('verdict', '?')}")
        return {
            "viability_score": total,
            "synthesis": updated_synthesis,
        }

    except Exception as e:
        error_msg = f"evaluator: scoring failed — {e}"
        print(f"[evaluator] Error: {error_msg}")
        return {
            "viability_score": None,
            "errors": [error_msg],
        }