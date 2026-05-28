# -*- coding: utf-8 -*-
"""
RiskAgent — orquestra as 4 APIs do Fintech Risk Framework
e gera relatorio em linguagem natural via Claude ou Ollama.
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Optional
import httpx
import anthropic


# Enderecos das 4 APIs do Fintech Risk Framework
API_FRAUD       = os.getenv("API_FRAUD",       "http://localhost:8000")
API_SEGMENTATION= os.getenv("API_SEGMENTATION","http://localhost:8001")
API_CREDIT      = os.getenv("API_CREDIT",      "http://localhost:8002")
API_DEFAULT     = os.getenv("API_DEFAULT",     "http://localhost:8003")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class ClientProfile:
    """Perfil unificado do cliente para analise de risco."""
    client_id: str
    # Dados de transacao (Fraud Detection)
    transaction_time: float
    transaction_amount: float
    v_features: list[float]          # V1..V28
    # Dados de segmentacao (Customer Segmentation)
    recency: float
    frequency: float
    monetary: float
    # Dados de credito (Credit Score + Default Prediction)
    age: int
    monthly_income: float
    dti: float
    fico_range_low: float
    revolving_utilization: float
    open_acc: int
    late_90_days: int
    real_estate_loans: int
    late_60_89_days: int
    late_30_59_days: int
    dependents: int
    # Dados de emprestimo (Default Prediction)
    loan_amnt: float
    int_rate: float
    grade: str
    emp_length_years: float
    revol_util: float
    total_acc: int
    inq_last_6mths: int
    pub_rec: int
    term_months: int


@dataclass
class RiskReport:
    """Relatorio consolidado de risco gerado pelo agente."""
    client_id: str
    fraud_result: dict
    segmentation_result: dict
    credit_result: dict
    default_result: dict
    narrative: str
    risk_summary: str
    latency_ms: float
    llm_provider: str


class RiskAgent:
    """
    Agente principal que orquestra as 4 APIs e gera relatorio.
    Usa Claude API com fallback automatico para Ollama local.
    """

    # Inicializa o agente com cliente HTTP e configuracoes de LLM
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=30.0)
        self._anthropic = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

    # Executa a analise completa de risco para um cliente
    async def analyze(self, profile: ClientProfile) -> RiskReport:
        start = time.perf_counter()

        # Chama as 4 APIs em paralelo para minimizar latencia
        fraud, seg, credit, default = await asyncio.gather(
            self._call_fraud(profile),
            self._call_segmentation(profile),
            self._call_credit(profile),
            self._call_default(profile),
            return_exceptions=True,
        )

        # Trata erros de APIs indisponiveis
        fraud   = fraud   if isinstance(fraud,   dict) else {"error": str(fraud)}
        seg     = seg     if isinstance(seg,     dict) else {"error": str(seg)}
        credit  = credit  if isinstance(credit,  dict) else {"error": str(credit)}
        default = default if isinstance(default, dict) else {"error": str(default)}

        # Gera narrativa com LLM
        narrative, provider = await self._generate_narrative(
            profile, fraud, seg, credit, default
        )

        risk_summary = self._compute_risk_summary(fraud, credit, default)
        latency_ms = (time.perf_counter() - start) * 1000

        return RiskReport(
            client_id=profile.client_id,
            fraud_result=fraud,
            segmentation_result=seg,
            credit_result=credit,
            default_result=default,
            narrative=narrative,
            risk_summary=risk_summary,
            latency_ms=round(latency_ms, 2),
            llm_provider=provider,
        )

    # Chama a API de Fraud Detection com os dados da transacao
    async def _call_fraud(self, p: ClientProfile) -> dict:
        payload = {
            "Time": p.transaction_time,
            "Amount": p.transaction_amount,
            **{f"V{i+1}": v for i, v in enumerate(p.v_features[:28])},
        }
        r = await self._http.post(f"{API_FRAUD}/predict", json=payload)
        return r.json()

    # Chama a API de Customer Segmentation com os dados RFM
    async def _call_segmentation(self, p: ClientProfile) -> dict:
        payload = {
            "customer_id": p.client_id,
            "recency": p.recency,
            "frequency": p.frequency,
            "monetary": p.monetary,
        }
        r = await self._http.post(f"{API_SEGMENTATION}/segment", json=payload)
        return r.json()

    # Chama a API de Credit Score com os dados financeiros
    async def _call_credit(self, p: ClientProfile) -> dict:
        payload = {
            "applicant_id": p.client_id,
            "RevolvingUtilizationOfUnsecuredLines": p.revolving_utilization,
            "age": p.age,
            "NumberOfTime30-59DaysPastDueNotWorse": p.late_30_59_days,
            "DebtRatio": p.dti,
            "MonthlyIncome": p.monthly_income,
            "NumberOfOpenCreditLinesAndLoans": p.open_acc,
            "NumberOfTimes90DaysLate": p.late_90_days,
            "NumberRealEstateLoansOrLines": p.real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": p.late_60_89_days,
            "NumberOfDependents": p.dependents,
        }
        r = await self._http.post(f"{API_CREDIT}/score", json=payload)
        return r.json()

    # Chama a API de Default Prediction com os dados do emprestimo
    async def _call_default(self, p: ClientProfile) -> dict:
        payload = {
            "loan_id": p.client_id,
            "loan_amnt": p.loan_amnt,
            "int_rate": p.int_rate,
            "grade": p.grade,
            "emp_length_years": p.emp_length_years,
            "annual_inc": p.monthly_income * 12,
            "dti": p.dti,
            "fico_range_low": p.fico_range_low,
            "open_acc": p.open_acc,
            "revol_util": p.revol_util,
            "total_acc": p.total_acc,
            "inq_last_6mths": p.inq_last_6mths,
            "pub_rec": p.pub_rec,
            "term_months": p.term_months,
        }
        r = await self._http.post(f"{API_DEFAULT}/predict", json=payload)
        return r.json()

    # Gera narrativa em linguagem natural via Claude com fallback para Ollama
    async def _generate_narrative(
        self, profile: ClientProfile,
        fraud: dict, seg: dict, credit: dict, default: dict
    ) -> tuple[str, str]:

        prompt = self._build_prompt(profile, fraud, seg, credit, default)

        # Tenta Claude API primeiro
        if self._anthropic and ANTHROPIC_KEY:
            try:
                return await asyncio.to_thread(
                    self._call_claude, prompt
                ), "claude"
            except Exception as e:
                print(f"Claude API falhou: {e} — usando Ollama")

        # Fallback para Ollama local
        try:
            return await self._call_ollama(prompt), "ollama"
        except Exception as e:
            return f"Erro ao gerar narrativa: {e}", "none"

    # Chama a API Claude para gerar o relatorio
    def _call_claude(self, prompt: str) -> str:
        msg = self._anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    # Chama o Ollama local como fallback
    async def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
        }
        r = await self._http.post(f"{OLLAMA_URL}/api/generate", json=payload)
        return r.json().get("response", "Sem resposta do Ollama")

    # Monta o prompt com todos os resultados para o LLM
    def _build_prompt(
        self, profile: ClientProfile,
        fraud: dict, seg: dict, credit: dict, default: dict
    ) -> str:
        return f"""Voce e um analista de risco financeiro senior. 
Analise os resultados abaixo e gere um relatorio executivo em portugues 
claro e objetivo para o cliente {profile.client_id}.

DETECCAO DE FRAUDE:
{json.dumps(fraud, indent=2, ensure_ascii=False)}

SEGMENTACAO DE CLIENTE:
{json.dumps(seg, indent=2, ensure_ascii=False)}

CREDIT SCORE:
{json.dumps(credit, indent=2, ensure_ascii=False)}

PREDICAO DE DEFAULT:
{json.dumps(default, indent=2, ensure_ascii=False)}

Gere um relatorio com:
1. Resumo executivo (2-3 linhas)
2. Perfil de risco consolidado
3. Pontos de atencao
4. Recomendacao final (APROVAR / REVISAR / REJEITAR)

Seja direto e use linguagem de negocio, nao tecnica."""

    # Calcula um resumo de risco consolidado baseado nos resultados das APIs
    def _compute_risk_summary(self, fraud: dict, credit: dict, default: dict) -> str:
        risk_score = 0

        if fraud.get("is_fraud"):
            risk_score += 40
        elif fraud.get("risk_label") == "HIGH":
            risk_score += 20
        elif fraud.get("risk_label") == "MEDIUM":
            risk_score += 10

        grade = credit.get("risk_grade", "C")
        grade_scores = {"A": 0, "B": 5, "C": 15, "D": 25, "E": 35}
        risk_score += grade_scores.get(grade, 20)

        tier = default.get("risk_tier", "MEDIUM")
        tier_scores = {"LOW": 0, "MEDIUM": 10, "HIGH": 20, "VERY_HIGH": 35}
        risk_score += tier_scores.get(tier, 10)

        if risk_score >= 60:   return "ALTO RISCO"
        if risk_score >= 30:   return "RISCO MODERADO"
        return "BAIXO RISCO"

    # Fecha o cliente HTTP ao encerrar o agente
    async def close(self) -> None:
        await self._http.aclose()
