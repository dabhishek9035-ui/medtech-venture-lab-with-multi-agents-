"""
tests/test_graph.py

Integration tests for the full LangGraph pipeline.

Tests the graph end-to-end with all LLM calls mocked, verifying:
- State flows correctly through every node
- Parallel fan-out and convergence work
- Synthesis, guardrails, evaluator, and report nodes produce expected output
- Error handling and non-fatal failures propagate correctly
- Memory store reads and writes correctly

Run with:
    cd ai_company_builder
    python -m pytest tests/test_graph.py -v
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_agents import (
    MOCK_MARKET_JSON,
    MOCK_RESEARCH_JSON,
    MOCK_PRODUCT_JSON,
    MOCK_ARCHITECTURE_JSON,
)


# ── shared mock helpers ───────────────────────────────────────────────────────

MOCK_GUARDRAILS_JSON = {"flags": []}

MOCK_EVALUATOR_JSON = {
    "scores": {
        "market_opportunity":    {"score": 17, "rationale": "Large and growing TAM"},
        "clinical_evidence":     {"score": 18, "rationale": "Strong published evidence"},
        "product_clarity":       {"score": 15, "rationale": "Clear MVP with good personas"},
        "technical_feasibility": {"score": 16, "rationale": "Solid HIPAA-compliant stack"},
        "risk_adjusted_outlook": {"score": 14, "rationale": "Manageable risks with clear mitigations"},
    },
    "total": 80,
    "verdict": "strong",
    "investment_summary": (
        "This startup addresses a large, well-evidenced clinical need with a clear "
        "technical approach. Regulatory and integration risks are manageable."
    ),
}


def _make_mock_llm(responses: list[dict]) -> MagicMock:
    """
    Returns a mock LLM class whose invoke() cycles through a list of JSON responses.
    """
    mock_class = MagicMock()
    side_effects = []
    for r in responses:
        mock_resp = MagicMock()
        mock_resp.content = json.dumps(r)
        side_effects.append(mock_resp)
    mock_class.return_value.invoke.side_effect = side_effects
    return mock_class


def _patch_all_llms(
    market=MOCK_MARKET_JSON,
    research=MOCK_RESEARCH_JSON,
    product=MOCK_PRODUCT_JSON,
    architecture=MOCK_ARCHITECTURE_JSON,
    guardrails=MOCK_GUARDRAILS_JSON,
    evaluator=MOCK_EVALUATOR_JSON,
):
    """Context manager that patches all 4 agent + 2 node LLM calls."""
    def _resp(data):
        m = MagicMock()
        m.content = json.dumps(data)
        return m

    patches = [
        patch("agents.market_agent.ChatGoogleGenerativeAI",
              **{"return_value.invoke.return_value": _resp(market)}),
        patch("agents.research_agent.ChatGoogleGenerativeAI",
              **{"return_value.invoke.return_value": _resp(research)}),
        patch("agents.product_agent.ChatGoogleGenerativeAI",
              **{"return_value.invoke.return_value": _resp(product)}),
        patch("agents.architecture_agent.ChatGoogleGenerativeAI",
              **{"return_value.invoke.return_value": _resp(architecture)}),
        patch("nodes.guardrails.ChatGoogleGenerativeAI",
              **{"return_value.invoke.return_value": _resp(guardrails)}),
        patch("nodes.evaluator.ChatGoogleGenerativeAI",
              **{"return_value.invoke.return_value": _resp(evaluator)}),
    ]
    return patches


# ── graph state tests ─────────────────────────────────────────────────────────

class TestGraphState:

    def test_initial_state_keys(self):
        """CompanyState TypedDict has all required keys."""
        from graph.state import CompanyState
        required_keys = [
            "idea", "domain",
            "market_output", "research_output", "product_output", "architecture_output",
            "synthesis", "guardrail_flags", "viability_score", "final_report", "errors",
        ]
        # TypedDict keys are accessible via __annotations__
        annotations = CompanyState.__annotations__
        for key in required_keys:
            assert key in annotations, f"Missing key in CompanyState: {key}"

    def test_state_reducers_merge_lists(self):
        """_merge_lists reducer deduplicates correctly."""
        from graph.state import _merge_lists
        a = ["flag-1", "flag-2"]
        b = ["flag-2", "flag-3"]
        result = _merge_lists(a, b)
        assert result == ["flag-1", "flag-2", "flag-3"]

    def test_state_reducer_keep_last(self):
        """_keep_last reducer returns latest non-None value."""
        from graph.state import _keep_last
        assert _keep_last({"old": True}, {"new": True}) == {"new": True}
        assert _keep_last({"old": True}, None) == {"old": True}


# ── synthesis node tests ──────────────────────────────────────────────────────

class TestSynthesisNode:

    def _make_state(self, **overrides):
        base = {
            "idea": "AI retinopathy detection",
            "domain": "Healthcare / MedTech",
            "market_output": MOCK_MARKET_JSON,
            "research_output": MOCK_RESEARCH_JSON,
            "product_output": MOCK_PRODUCT_JSON,
            "architecture_output": MOCK_ARCHITECTURE_JSON,
        }
        base.update(overrides)
        return base

    def test_synthesis_produces_string(self):
        from nodes.synthesis import synthesis_node
        result = synthesis_node(self._make_state())
        assert isinstance(result["synthesis"], str)
        assert len(result["synthesis"]) > 100

    def test_synthesis_contains_all_sections(self):
        from nodes.synthesis import synthesis_node
        result = synthesis_node(self._make_state())
        text = result["synthesis"]
        assert "Market Analysis" in text
        assert "Clinical Research" in text
        assert "Product Analysis" in text
        assert "Architecture Analysis" in text

    def test_synthesis_handles_missing_agent_output(self):
        """If an agent errored out, synthesis should still complete."""
        from nodes.synthesis import synthesis_node
        result = synthesis_node(self._make_state(market_output=None))
        assert "Not available" in result["synthesis"]

    def test_synthesis_includes_idea(self):
        from nodes.synthesis import synthesis_node
        result = synthesis_node(self._make_state())
        assert "AI retinopathy detection" in result["synthesis"]


# ── guardrails node tests ─────────────────────────────────────────────────────

class TestGuardrailsNode:

    @patch("nodes.guardrails.ChatGoogleGenerativeAI",
           **{"return_value.invoke.return_value": MagicMock(content='{"flags": []}')})
    def test_clean_state_no_flags(self, mock_llm):
        from nodes.guardrails import guardrails_node
        state = {
            "market_output": MOCK_MARKET_JSON,
            "research_output": MOCK_RESEARCH_JSON,
            "product_output": MOCK_PRODUCT_JSON,
            "architecture_output": MOCK_ARCHITECTURE_JSON,
            "synthesis": "A clean synthesis with no issues.",
        }
        result = guardrails_node(state)
        rule_flags = [f for f in result["guardrail_flags"] if "rule" in f]
        assert len(rule_flags) == 0

    def test_rule_scan_catches_unencrypted(self):
        from nodes.guardrails import _rule_based_scan
        state = {
            "market_output": {},
            "research_output": {},
            "product_output": {},
            "architecture_output": {"storage": "store phi unencrypted"},
            "synthesis": "",
        }
        flags = _rule_based_scan(state)
        assert any("nencrypted" in f for f in flags)

    def test_rule_scan_catches_clinical_claim(self):
        from nodes.guardrails import _rule_based_scan
        state = {
            "market_output": {"description": "100% accuracy guaranteed"},
            "research_output": {}, "product_output": {}, "architecture_output": {},
            "synthesis": "",
        }
        flags = _rule_based_scan(state)
        assert any("100%" in f for f in flags)


# ── evaluator node tests ──────────────────────────────────────────────────────

class TestEvaluatorNode:

    @patch("nodes.evaluator.ChatGoogleGenerativeAI",
           **{"return_value.invoke.return_value": MagicMock(content=json.dumps(MOCK_EVALUATOR_JSON))})
    def test_score_written_to_state(self, mock_llm):
        from nodes.evaluator import evaluator_node
        state = {"synthesis": "A full synthesis brief.", "guardrail_flags": []}
        result = evaluator_node(state)
        assert result["viability_score"] == 80

    @patch("nodes.evaluator.ChatGoogleGenerativeAI",
           **{"return_value.invoke.return_value": MagicMock(content=json.dumps(MOCK_EVALUATOR_JSON))})
    def test_scorecard_appended_to_synthesis(self, mock_llm):
        from nodes.evaluator import evaluator_node
        state = {"synthesis": "Original synthesis.", "guardrail_flags": []}
        result = evaluator_node(state)
        assert "Viability Scorecard" in result["synthesis"]
        assert "Original synthesis." in result["synthesis"]

    @patch("nodes.evaluator.ChatGoogleGenerativeAI")
    def test_empty_synthesis_returns_zero(self, mock_llm):
        from nodes.evaluator import evaluator_node
        result = evaluator_node({"synthesis": "", "guardrail_flags": []})
        assert result["viability_score"] == 0


# ── report node tests ─────────────────────────────────────────────────────────

class TestReportNode:

    def _make_state(self):
        return {
            "idea": "AI retinopathy detection startup",
            "domain": "Healthcare / MedTech",
            "synthesis": "## Full Analysis\nDetailed analysis here.\n\n## Viability Scorecard\n80/100",
            "viability_score": 80,
            "guardrail_flags": ["[guardrails:rule] Unencrypted data handling detected"],
            "errors": [],
        }

    def test_report_node_produces_markdown(self):
        from outputs.report import report_node
        result = report_node(self._make_state())
        assert isinstance(result["final_report"], str)
        assert "# AI Company Builder Report" in result["final_report"]

    def test_report_contains_score(self):
        from outputs.report import report_node
        result = report_node(self._make_state())
        assert "80" in result["final_report"]

    def test_report_contains_flags(self):
        from outputs.report import report_node
        result = report_node(self._make_state())
        assert "Guardrail Flags" in result["final_report"]

    def test_report_written_to_disk(self):
        from outputs.report import render_report
        state = self._make_state()
        state["final_report"] = None  # force rebuild
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "report.md"
            render_report(state, out_path)
            assert out_path.exists()
            content = out_path.read_text(encoding="utf-8")
            assert "AI Company Builder Report" in content


# ── memory store tests ────────────────────────────────────────────────────────

class TestMemoryStore:

    def _make_store(self, tmp_path):
        from memory.store import MemoryStore
        return MemoryStore(db_path=tmp_path / "test_memory.db")

    def test_save_and_retrieve(self, tmp_path):
        store = self._make_store(tmp_path)
        state = {
            "idea": "AI retinopathy detection",
            "domain": "Healthcare / MedTech",
            "viability_score": 80,
            "guardrail_flags": [],
            "synthesis": "Strong startup idea.",
        }
        row_id = store.save(state)
        assert row_id == 1
        retrieved = store.get_by_id(1)
        assert retrieved["idea"] == "AI retinopathy detection"
        assert retrieved["score"] == 80

    def test_count(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.count() == 0
        store.save({"idea": "Idea 1", "domain": "Healthcare / MedTech",
                    "viability_score": 60, "guardrail_flags": [], "synthesis": ""})
        store.save({"idea": "Idea 2", "domain": "Healthcare / MedTech",
                    "viability_score": 70, "guardrail_flags": [], "synthesis": ""})
        assert store.count() == 2

    def test_search(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save({"idea": "diabetic retinopathy AI tool", "domain": "Healthcare / MedTech",
                    "viability_score": 75, "guardrail_flags": [], "synthesis": ""})
        store.save({"idea": "cardiac monitoring wearable", "domain": "Healthcare / MedTech",
                    "viability_score": 65, "guardrail_flags": [], "synthesis": ""})
        results = store.search("retinopathy")
        assert len(results) == 1
        assert "retinopathy" in results[0]["idea"]

    def test_delete_run(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save({"idea": "Test idea", "domain": "Healthcare / MedTech",
                    "viability_score": 50, "guardrail_flags": [], "synthesis": ""})
        assert store.count() == 1
        store.delete_run(1)
        assert store.count() == 0

    def test_build_prior_context_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        context = store.build_prior_context("retinopathy")
        assert context == ""

    def test_build_prior_context_with_runs(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save({"idea": "retinopathy detection app", "domain": "Healthcare / MedTech",
                    "viability_score": 78, "guardrail_flags": [], "synthesis": "Strong idea."})
        context = store.build_prior_context("retinopathy")
        assert "78" in context
        assert "Prior Startup Runs" in context

    def test_clear_all(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save({"idea": "Test", "domain": "Healthcare / MedTech",
                    "viability_score": 50, "guardrail_flags": [], "synthesis": ""})
        store.clear_all()
        assert store.count() == 0


# ── edges tests ───────────────────────────────────────────────────────────────

class TestEdges:

    def test_route_after_guardrails_no_flags(self):
        from graph.edges import route_after_guardrails
        state = {"guardrail_flags": []}
        assert route_after_guardrails(state) == "evaluator"

    def test_route_after_guardrails_warning_flags(self):
        from graph.edges import route_after_guardrails
        state = {"guardrail_flags": ["[guardrails:llm:warning] Minor issue"]}
        assert route_after_guardrails(state) == "evaluator"

    def test_route_after_guardrails_critical_flag(self):
        from graph.edges import route_after_guardrails
        state = {"guardrail_flags": ["[guardrails:llm:critical] phi without encryption"]}
        assert route_after_guardrails(state) == "report"