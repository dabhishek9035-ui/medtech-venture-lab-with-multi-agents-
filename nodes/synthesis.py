"""
nodes/synthesis.py

Synthesis node — the first convergence point after the 4 parallel agents.

Reads all 4 agent output dicts from state and merges them into a single
coherent narrative string written to state["synthesis"].

Does NOT call Gemini — synthesis is done in pure Python by templating
the structured agent outputs into a readable brief. This keeps the
synthesis step fast, deterministic, and free of token cost.

The synthesis string is consumed by:
- nodes/evaluator.py  (passed as context for scoring)
- output/report.py    (embedded in the final Markdown report)
"""

from __future__ import annotations

from graph.state import CompanyState


# ── section formatters ────────────────────────────────────────────────────────

def _fmt_market(m: dict | None) -> str:
    if not m:
        return "**Market Analysis**: Not available (agent error).\n"

    competitors = m.get("competitors") or []
    comp_lines = "\n".join(
        f"  - {c.get('name', 'Unknown')} ({c.get('stage', '?')}): weakness — {c.get('weakness', '?')}"
        for c in competitors
    ) or "  - None identified"

    risks = "\n".join(f"  - {r}" for r in (m.get("market_risks") or []))
    tailwinds = "\n".join(f"  - {t}" for t in (m.get("market_tailwinds") or []))

    reg = m.get("regulatory_landscape") or {}

    return (
        "**Market Analysis**\n"
        f"- TAM: {m.get('tam', 'Unknown')}\n"
        f"- SAM: {m.get('sam', 'Unknown')}\n"
        f"- Regulatory pathway: {reg.get('primary_pathway', 'Unknown')} "
        f"(est. {reg.get('estimated_timeline', '?')})\n"
        f"- Competitors:\n{comp_lines}\n"
        f"- Market risks:\n{risks}\n"
        f"- Tailwinds:\n{tailwinds}\n"
        f"- GTM: {m.get('go_to_market', 'Not specified')}\n"
    )


def _fmt_research(r: dict | None) -> str:
    if not r:
        return "**Clinical Research**: Not available (agent error).\n"

    evidence = r.get("clinical_evidence") or {}
    studies = evidence.get("key_studies") or []
    study_lines = "\n".join(
        f"  - {s.get('title', '?')} ({s.get('year', '?')}): {s.get('finding', '?')}"
        for s in studies
    ) or "  - No key studies identified"

    gaps = "\n".join(f"  - {g}" for g in (r.get("research_gaps") or []))
    challenges = "\n".join(f"  - {c}" for c in (r.get("key_challenges") or []))

    benchmarks = r.get("benchmarks") or {}
    perf_targets = "\n".join(
        f"  - {t}" for t in (benchmarks.get("performance_targets") or [])
    )

    return (
        "**Clinical Research**\n"
        f"- Evidence strength: {evidence.get('strength', 'Unknown')}\n"
        f"- Summary: {evidence.get('summary', 'Not available')}\n"
        f"- Key studies:\n{study_lines}\n"
        f"- Performance targets:\n{perf_targets}\n"
        f"- Research gaps:\n{gaps}\n"
        f"- Key challenges:\n{challenges}\n"
        f"- Recommended study: {r.get('recommended_study_design', 'Not specified')}\n"
        f"- IP landscape: {r.get('ip_landscape', 'Not assessed')}\n"
    )


def _fmt_product(p: dict | None) -> str:
    if not p:
        return "**Product Analysis**: Not available (agent error).\n"

    features = p.get("mvp_features") or []
    feature_lines = "\n".join(
        f"  - [{f.get('priority', '?').upper()}] {f.get('feature', '?')}: {f.get('rationale', '?')}"
        for f in features
    ) or "  - No features defined"

    personas = p.get("user_personas") or []
    persona_lines = "\n".join(
        f"  - {pe.get('name', '?')} — pain: {pe.get('pain_point', '?')} | barrier: {pe.get('adoption_barrier', '?')}"
        for pe in personas
    ) or "  - No personas defined"

    ux = "\n".join(f"  {step}" for step in (p.get("ux_flow") or []))
    diffs = "\n".join(f"  - {d}" for d in (p.get("differentiators") or []))

    metrics = p.get("success_metrics") or []
    metric_lines = "\n".join(
        f"  - {me.get('metric', '?')}: target {me.get('target', '?')} ({me.get('measurement', '?')})"
        for me in metrics
    )

    return (
        "**Product Analysis**\n"
        f"- MVP timeline: {p.get('mvp_timeline_weeks', '?')} weeks\n"
        f"- MVP features:\n{feature_lines}\n"
        f"- User personas:\n{persona_lines}\n"
        f"- UX flow:\n{ux}\n"
        f"- Differentiators:\n{diffs}\n"
        f"- Success metrics:\n{metric_lines}\n"
        f"- Biggest product risk: {p.get('biggest_product_risk', 'Not identified')}\n"
    )


def _fmt_architecture(a: dict | None) -> str:
    if not a:
        return "**Architecture Analysis**: Not available (agent error).\n"

    stack = a.get("tech_stack") or {}
    pipeline = a.get("data_pipeline") or {}
    hipaa = a.get("hipaa_considerations") or []
    hipaa_lines = "\n".join(
        f"  - {h.get('requirement', '?')}: {h.get('implementation', '?')}"
        for h in hipaa
    ) or "  - Not specified"

    integrations = "\n".join(f"  - {i}" for i in (a.get("integration_points") or []))

    return (
        "**Architecture Analysis**\n"
        f"- ML framework: {stack.get('ml_framework', '?')}\n"
        f"- Backend: {stack.get('backend', '?')}\n"
        f"- Frontend: {stack.get('frontend', '?')}\n"
        f"- Cloud: {stack.get('cloud_provider', '?')}\n"
        f"- Database: {stack.get('database', '?')}\n"
        f"- Model serving: {stack.get('model_serving', '?')}\n"
        f"- Data ingestion: {pipeline.get('ingestion', '?')}\n"
        f"- Storage: {pipeline.get('storage', '?')}\n"
        f"- Versioning: {pipeline.get('versioning', '?')}\n"
        f"- HIPAA requirements:\n{hipaa_lines}\n"
        f"- Scalability: {a.get('scalability', 'Not specified')}\n"
        f"- Integrations:\n{integrations}\n"
        f"- MVP infra cost: {a.get('mvp_infra_cost_monthly_usd', '?')}/month\n"
        f"- Biggest technical risk: {a.get('biggest_technical_risk', 'Not identified')}\n"
    )


# ── node ──────────────────────────────────────────────────────────────────────

def synthesis_node(state: CompanyState) -> dict:
    """
    LangGraph node for synthesis.
    Merges all 4 agent outputs into a single structured narrative string.
    """
    print("[synthesis] Merging agent outputs...")

    idea = state.get("idea", "No idea provided")
    domain = state.get("domain", "Healthcare / MedTech")

    sections = [
        f"# Startup Brief\n",
        f"**Domain**: {domain}\n",
        f"**Idea**: {idea}\n",
        "---\n",
        _fmt_market(state.get("market_output")),
        "---\n",
        _fmt_research(state.get("research_output")),
        "---\n",
        _fmt_product(state.get("product_output")),
        "---\n",
        _fmt_architecture(state.get("architecture_output")),
    ]

    synthesis_text = "\n".join(sections)

    print(f"[synthesis] Done. Brief is {len(synthesis_text)} chars.")
    return {"synthesis": synthesis_text}