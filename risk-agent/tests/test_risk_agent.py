# -*- coding: utf-8 -*-
"""Testes do Risk Agent com mocks das 4 APIs."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from api.main import app
from agent.risk_agent import RiskAgent, ClientProfile, RiskReport


def make_profile(**kwargs):
    defaults = dict(
        client_id="TEST001",
        transaction_time=406.0, transaction_amount=150.0,
        v_features=[0.0]*28,
        recency=30.0, frequency=10.0, monetary=5000.0,
        age=35, monthly_income=8000.0, dti=0.3,
        fico_range_low=680.0, revolving_utilization=0.4,
        open_acc=8, late_90_days=0, real_estate_loans=1,
        late_60_89_days=0, late_30_59_days=0, dependents=1,
        loan_amnt=15000.0, int_rate=12.5, grade="C",
        emp_length_years=5.0, revol_util=40.0, total_acc=20,
        inq_last_6mths=1, pub_rec=0, term_months=36,
    )
    defaults.update(kwargs)
    return ClientProfile(**defaults)


# Resultados simulados das 4 APIs
MOCK_FRAUD   = {"is_fraud": False, "fraud_probability": 0.05, "risk_label": "LOW", "latency_ms": 12.0}
MOCK_SEG     = {"segment_label": "Champions", "rfm_score": 0.82, "cluster_id": 0, "is_high_value": True}
MOCK_CREDIT  = {"score": 750, "risk_grade": "B", "recommendation": "APPROVE", "pd_12m": 0.04}
MOCK_DEFAULT = {"survival_at_12m": 0.96, "pd_12m": 0.04, "risk_tier": "LOW", "hazard_ratio": 0.8}


def test_client_profile_creation():
    profile = make_profile()
    assert profile.client_id == "TEST001"
    assert profile.age == 35
    assert len(profile.v_features) == 28


def test_risk_summary_baixo():
    agent = RiskAgent.__new__(RiskAgent)
    summary = agent._compute_risk_summary(MOCK_FRAUD, MOCK_CREDIT, MOCK_DEFAULT)
    assert summary == "BAIXO RISCO"


def test_risk_summary_alto():
    agent = RiskAgent.__new__(RiskAgent)
    fraud_alto = {"is_fraud": True, "risk_label": "HIGH"}
    credit_alto = {"risk_grade": "E", "recommendation": "DENY"}
    default_alto = {"risk_tier": "VERY_HIGH", "pd_12m": 0.45}
    summary = agent._compute_risk_summary(fraud_alto, credit_alto, default_alto)
    assert summary == "ALTO RISCO"


def test_build_prompt_contains_client_id():
    agent = RiskAgent.__new__(RiskAgent)
    profile = make_profile(client_id="CLIENTE_TESTE")
    prompt = agent._build_prompt(profile, MOCK_FRAUD, MOCK_SEG, MOCK_CREDIT, MOCK_DEFAULT)
    assert "CLIENTE_TESTE" in prompt
    assert "DETECCAO DE FRAUDE" in prompt
    assert "CREDIT SCORE" in prompt


@pytest.mark.asyncio
async def test_analyze_with_mocked_apis():
    agent = RiskAgent.__new__(RiskAgent)

    async def mock_fraud(p):    return MOCK_FRAUD
    async def mock_seg(p):      return MOCK_SEG
    async def mock_credit(p):   return MOCK_CREDIT
    async def mock_default(p):  return MOCK_DEFAULT
    async def mock_narrative(p, f, s, c, d): return "Relatorio de teste.", "claude"

    agent._call_fraud        = mock_fraud
    agent._call_segmentation = mock_seg
    agent._call_credit       = mock_credit
    agent._call_default      = mock_default
    agent._generate_narrative= mock_narrative

    profile = make_profile()
    report = await agent.analyze(profile)

    assert report.client_id == "TEST001"
    assert report.risk_summary == "BAIXO RISCO"
    assert report.llm_provider == "claude"
    assert report.latency_ms > 0


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_analyze_endpoint_with_mock():
    from unittest.mock import patch, AsyncMock
    mock_report = RiskReport(
        client_id="C001",
        fraud_result=MOCK_FRAUD,
        segmentation_result=MOCK_SEG,
        credit_result=MOCK_CREDIT,
        default_result=MOCK_DEFAULT,
        narrative="Cliente de baixo risco. Recomendado APROVAR.",
        risk_summary="BAIXO RISCO",
        latency_ms=450.0,
        llm_provider="claude",
    )
    with patch("api.main._agent") as mock_agent:
        mock_agent.analyze = AsyncMock(return_value=mock_report)
        client = TestClient(app)
        payload = {
            "client_id": "C001",
            "transaction_time": 406.0, "transaction_amount": 150.0,
            "v_features": [0.0]*28,
            "recency": 30.0, "frequency": 10.0, "monetary": 5000.0,
            "age": 35, "monthly_income": 8000.0, "dti": 0.3,
            "fico_range_low": 680.0, "revolving_utilization": 0.4,
            "open_acc": 8, "late_90_days": 0, "real_estate_loans": 1,
            "late_60_89_days": 0, "late_30_59_days": 0, "dependents": 1,
            "loan_amnt": 15000.0, "int_rate": 12.5, "grade": "C",
            "emp_length_years": 5.0, "revol_util": 40.0, "total_acc": 20,
            "inq_last_6mths": 1, "pub_rec": 0, "term_months": 36,
        }
        r = client.post("/analyze", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["risk_summary"] == "BAIXO RISCO"
        assert data["llm_provider"] == "claude"
