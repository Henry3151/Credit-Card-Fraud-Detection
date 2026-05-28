# -*- coding: utf-8 -*-
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.risk_agent import RiskAgent, ClientProfile

_agent: Optional[RiskAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = RiskAgent()
    yield
    await _agent.close()

app = FastAPI(
    title="Risk Agent API",
    description="Agente de analise de risco financeiro — Fintech Risk Framework",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    client_id: str = "C001"
    transaction_time: float = 406.0
    transaction_amount: float = 150.0
    v_features: list[float] = Field(default_factory=lambda: [0.0] * 28)
    recency: float = 30.0
    frequency: float = 10.0
    monetary: float = 5000.0
    age: int = 35
    monthly_income: float = 8000.0
    dti: float = 0.3
    fico_range_low: float = 680.0
    revolving_utilization: float = 0.4
    open_acc: int = 8
    late_90_days: int = 0
    real_estate_loans: int = 1
    late_60_89_days: int = 0
    late_30_59_days: int = 0
    dependents: int = 1
    loan_amnt: float = 15000.0
    int_rate: float = 12.5
    grade: str = "C"
    emp_length_years: float = 5.0
    revol_util: float = 40.0
    total_acc: int = 20
    inq_last_6mths: int = 1
    pub_rec: int = 0
    term_months: int = 36

class AnalyzeResponse(BaseModel):
    client_id: str
    fraud_result: dict
    segmentation_result: dict
    credit_result: dict
    default_result: dict
    narrative: str
    risk_summary: str
    latency_ms: float
    llm_provider: str

@app.get("/health")
async def health():
    return {"status": "ok", "service": "risk-agent"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agente nao inicializado")
    profile = ClientProfile(**req.model_dump())
    report = await _agent.analyze(profile)
    return AnalyzeResponse(
        client_id=report.client_id,
        fraud_result=report.fraud_result,
        segmentation_result=report.segmentation_result,
        credit_result=report.credit_result,
        default_result=report.default_result,
        narrative=report.narrative,
        risk_summary=report.risk_summary,
        latency_ms=report.latency_ms,
        llm_provider=report.llm_provider,
    )

ui_path = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(ui_path, "index.html"))
