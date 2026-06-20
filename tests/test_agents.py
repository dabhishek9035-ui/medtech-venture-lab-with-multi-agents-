"""
tests/test_agents.py

Unit tests for individual agent nodes.

Tests run WITHOUT calling the Gemini API — all LLM calls are mocked
using unittest.mock.patch so tests are:
- Free (no API quota consumed)
- Fast (no network calls)
- Deterministic (no model variance)

Run with:
    cd ai_company_builder
    python -m pytest tests/test_agents.py -v
"""

import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_IDEA = (
    "Domain: Healthcare / MedTech\n"
    "Startup idea: AI-powered early detection of diabetic retinopathy "
    "using fundus photographs analysed by a deep learning model.\n\n"
    "Analyse this idea through the lens of a Healthcare / MedTech startup."
)

MOCK_MARKET_JSON = {
    "tam": "$12B global diabetic retinopathy screening market by 2028",
    "sam": "$1.2B addressable in US + India tier-1 hospitals",
    "competitors": [
        {"name": "IDx-DR", "stage": "public", "weakness": "Limited to retinopathy only"},
        {"name": "Eyenuk", "stage": "growth", "weakness": "High per-scan cost"},
    ],
    "regulatory_landscape": {
        "primary_pathway": "FDA De Novo",
        "estimated_timeline": "18-24 months",
        "key_requirements": ["Clinical validation study", "Software as Medical Device (SaMD) classification"],
    },
    "market_risks": ["Reimbursement uncertainty", "EHR integration complexity"],
    "market_tailwinds": ["Growing diabetic population", "Telemedicine adoption"],
    "go_to_market": "Target large ophthalmology chains and diabetic clinics in India and US.",
}

MOCK_RESEARCH_JSON = {
    "clinical_evidence": {
        "strength": "strong",
        "summary": "Multiple large-scale studies validate DL models for retinopathy detection.",
        "key_studies": [
            {"title": "Gulshan et al. JAMA 2016", "finding": "AUC 0.99 on EyePACS dataset", "year": "2016"},
        ],
    },
    "benchmarks": {
        "current_standard_of_care": "Manual grading by ophthalmologist",
        "performance_targets": ["Sensitivity >= 90%", "Specificity >= 85%"],
        "validation_datasets": ["EyePACS", "Messidor-2"],
    },
    "research_gaps": ["Real-world performance in low-resource settings"],
    "key_challenges": ["Dataset diversity", "Regulatory approval timeline"],
    "recommended_study_design": "Prospective multi-site study across 3 hospital systems",
    "ip_landscape": "Several patents held by Google Health; freedom-to-operate analysis recommended.",
}

MOCK_PRODUCT_JSON = {
    "mvp_features": [
        {"feature": "Fundus image upload", "priority": "must-have", "rationale": "Core input"},
        {"feature": "AI grading result", "priority": "must-have", "rationale": "Core output"},
    ],
    "user_personas": [
        {"name": "Dr. Priya, Ophthalmologist", "role": "Ophthalmologist",
         "pain_point": "Too many patients, too little time", "adoption_barrier": "Trust in AI accuracy"},
    ],
    "ux_flow": ["Step 1: Upload fundus image", "Step 2: Receive AI grade", "Step 3: Review report"],
    "differentiators": ["Offline mode for low-connectivity clinics", "Explainable AI overlay"],
    "mvp_timeline_weeks": "16",
    "success_metrics": [
        {"metric": "Sensitivity", "target": ">=90%", "measurement": "Clinical validation study"},
    ],
    "biggest_product_risk": "Clinician trust in AI recommendations without explainability.",
}

MOCK_ARCHITECTURE_JSON = {
    "tech_stack": {
        "ml_framework": "PyTorch",
        "backend": "FastAPI + Python 3.11",
        "frontend": "React + TypeScript",
        "cloud_provider": "AWS (HIPAA BAA available)",
        "database": "PostgreSQL + pgvector",
        "model_serving": "AWS SageMaker",
    },
    "data_pipeline": {
        "ingestion": "DICOM via HL7 FHIR API",
        "preprocessing": "Deidentification via AWS Comprehend Medical",
        "storage": "S3 with AES-256 encryption at rest",
        "versioning": "DVC + MLflow",
    },
    "hipaa_considerations": [
        {"requirement": "Encryption at rest", "implementation": "AES-256 on S3 and RDS"},
        {"requirement": "Audit logging", "implementation": "AWS CloudTrail + custom audit table"},
    ],
    "scalability": "Horizontal scaling via ECS Fargate; SageMaker auto-scaling for inference.",
    "integration_points": ["HL7 FHIR R4", "Epic MyApps"],
    "mvp_infra_cost_monthly_usd": "$300-600/month for MVP stage on AWS",
    "biggest_technical_risk": "DICOM ingestion complexity from legacy hospital systems.",
}


def _mock_llm_response(json_data: dict) -> MagicMock:
    """Creates a mock LLM response object returning the given dict as JSON."""
    mock_response = MagicMock()
    mock_response.content = json.dumps(json_data)
    return mock_response


# ── orchestrator tests ────────────────────────────────────────────────────────

class TestOrchestratorNode:

    def test_valid_idea_passes(self):
        from agents.orchestrator import orchestrator_node
        state = {"idea": "AI startup for diabetic retinopathy detection", "domain": "Healthcare / MedTech"}
        result = orchestrator_node(state)
        assert "idea" in result
        assert result["errors"] == []

    def test_empty_idea_returns_error(self):
        from agents.orchestrator import orchestrator_node
        state = {"idea": "", "domain": "Healthcare / MedTech"}
        result = orchestrator_node(state)
        assert len(result["errors"]) > 0

    def test_short_idea_returns_error(self):
        from agents.orchestrator import orchestrator_node
        state = {"idea": "AI app", "domain": "Healthcare / MedTech"}
        result = orchestrator_node(state)
        assert any("too short" in e for e in result["errors"])

    def test_long_idea_is_truncated(self):
        from agents.orchestrator import orchestrator_node
        long_idea = "A" * 3000
        state = {"idea": long_idea, "domain": "Healthcare / MedTech"}
        result = orchestrator_node(state)
        assert any("truncated" in e for e in result["errors"])

    def test_framing_injected_into_idea(self):
        from agents.orchestrator import orchestrator_node
        state = {"idea": "AI startup for diabetic retinopathy detection", "domain": "Healthcare / MedTech"}
        result = orchestrator_node(state)
        assert "Domain:" in result["idea"]
        assert "Healthcare" in result["idea"]

    def test_unknown_domain_soft_warns(self):
        from agents.orchestrator import orchestrator_node
        state = {"idea": "AI startup for diabetic retinopathy detection", "domain": "FinTech"}
        result = orchestrator_node(state)
        assert any("not in the supported list" in e for e in result["errors"])


# ── market agent tests ────────────────────────────────────────────────────────

class TestMarketNode:

    @patch("agents.market_agent.ChatGoogleGenerativeAI")
    def test_successful_parse(self, mock_llm_class):
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(MOCK_MARKET_JSON)
        from agents.market_agent import market_node
        result = market_node({"idea": SAMPLE_IDEA})
        assert result["market_output"] is not None
        assert "tam" in result["market_output"]
        assert result["errors"] == []

    @patch("agents.market_agent.ChatGoogleGenerativeAI")
    def test_invalid_json_returns_error(self, mock_llm_class):
        mock_resp = MagicMock()
        mock_resp.content = "This is not JSON at all."
        mock_llm_class.return_value.invoke.return_value = mock_resp
        from agents.market_agent import market_node
        result = market_node({"idea": SAMPLE_IDEA})
        assert result["market_output"] is None
        assert len(result["errors"]) > 0

    @patch("agents.market_agent.ChatGoogleGenerativeAI")
    def test_llm_failure_returns_error(self, mock_llm_class):
        mock_llm_class.return_value.invoke.side_effect = Exception("API rate limit")
        from agents.market_agent import market_node
        result = market_node({"idea": SAMPLE_IDEA})
        assert result["market_output"] is None
        assert any("failed" in e for e in result["errors"])

    @patch("agents.market_agent.ChatGoogleGenerativeAI")
    def test_strips_markdown_fences(self, mock_llm_class):
        fenced = f"```json\n{json.dumps(MOCK_MARKET_JSON)}\n```"
        mock_resp = MagicMock()
        mock_resp.content = fenced
        mock_llm_class.return_value.invoke.return_value = mock_resp
        from agents.market_agent import market_node
        result = market_node({"idea": SAMPLE_IDEA})
        assert result["market_output"] is not None


# ── research agent tests ──────────────────────────────────────────────────────

class TestResearchNode:

    @patch("agents.research_agent.ChatGoogleGenerativeAI")
    def test_successful_parse(self, mock_llm_class):
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(MOCK_RESEARCH_JSON)
        from agents.research_agent import research_node
        result = research_node({"idea": SAMPLE_IDEA})
        assert result["research_output"] is not None
        assert "clinical_evidence" in result["research_output"]
        assert result["errors"] == []

    @patch("agents.research_agent.ChatGoogleGenerativeAI")
    def test_evidence_strength_present(self, mock_llm_class):
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(MOCK_RESEARCH_JSON)
        from agents.research_agent import research_node
        result = research_node({"idea": SAMPLE_IDEA})
        assert result["research_output"]["clinical_evidence"]["strength"] == "strong"


# ── product agent tests ───────────────────────────────────────────────────────

class TestProductNode:

    @patch("agents.product_agent.ChatGoogleGenerativeAI")
    def test_successful_parse(self, mock_llm_class):
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(MOCK_PRODUCT_JSON)
        from agents.product_agent import product_node
        result = product_node({"idea": SAMPLE_IDEA})
        assert result["product_output"] is not None
        assert "mvp_features" in result["product_output"]
        assert result["errors"] == []

    @patch("agents.product_agent.ChatGoogleGenerativeAI")
    def test_must_have_features_present(self, mock_llm_class):
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(MOCK_PRODUCT_JSON)
        from agents.product_agent import product_node
        result = product_node({"idea": SAMPLE_IDEA})
        priorities = [f["priority"] for f in result["product_output"]["mvp_features"]]
        assert "must-have" in priorities


# ── architecture agent tests ──────────────────────────────────────────────────

class TestArchitectureNode:

    @patch("agents.architecture_agent.ChatGoogleGenerativeAI")
    def test_successful_parse(self, mock_llm_class):
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(MOCK_ARCHITECTURE_JSON)
        from agents.architecture_agent import architecture_node
        result = architecture_node({"idea": SAMPLE_IDEA})
        assert result["architecture_output"] is not None
        assert "tech_stack" in result["architecture_output"]
        assert result["errors"] == []

    @patch("agents.architecture_agent.ChatGoogleGenerativeAI")
    def test_hipaa_flags_not_triggered_on_clean_output(self, mock_llm_class):
        """Clean architecture output should not trigger HIPAA flags."""
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(MOCK_ARCHITECTURE_JSON)
        from agents.architecture_agent import architecture_node
        result = architecture_node({"idea": SAMPLE_IDEA})
        # MOCK_ARCHITECTURE_JSON has AES-256 encryption — no unencrypted flag expected
        flags = result.get("guardrail_flags", [])
        assert not any("unencrypted" in f.lower() for f in flags)

    @patch("agents.architecture_agent.ChatGoogleGenerativeAI")
    def test_hipaa_flag_triggered_on_bad_output(self, mock_llm_class):
        """Architecture output mentioning unencrypted PHI should trigger a flag."""
        bad_arch = {**MOCK_ARCHITECTURE_JSON}
        bad_arch["data_pipeline"] = {**bad_arch["data_pipeline"], "storage": "store phi unencrypted in S3"}
        mock_llm_class.return_value.invoke.return_value = _mock_llm_response(bad_arch)
        from agents.architecture_agent import architecture_node
        result = architecture_node({"idea": SAMPLE_IDEA})
        flags = result.get("guardrail_flags", [])
        assert len(flags) > 0