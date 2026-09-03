"""FastAPI application for churn prediction and service diagnostics."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from customer_intelligence.churn import ChurnModelBundle, load_churn_bundle

from .config import get_settings
from .storage import DecisionRepository
from .schemas import (
    CustomerDecisionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the process starts, not on every request."""
    settings = get_settings()
    app.state.bundle = None
    app.state.model_load_error = None
    try:
        app.state.bundle = load_churn_bundle(settings.model_path)
    except Exception as exc:  # readiness reports failure without exposing detail
        app.state.model_load_error = str(exc)
    app.state.repository = None
    app.state.database_error = None
    try:
        repository = DecisionRepository(settings.database_url)
        repository.initialize()
        repository.ping()
        app.state.repository = repository
    except Exception as exc:
        app.state.database_error = str(exc)
    yield
    app.state.bundle = None
    app.state.repository = None


app = FastAPI(
    title="Customer Intelligence Churn API",
    version="1.0.0",
    description=(
        "Scores leakage-safe customer features with the approved churn model. "
        "Population campaign-priority bands are produced by the batch workflow."
    ),
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the stakeholder dashboard from the same origin as the API."""
    return FileResponse(STATIC_DIR / "index.html")


def require_bundle(request: Request) -> ChurnModelBundle:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready",
        )
    return bundle


def require_repository(request: Request) -> DecisionRepository:
    repository = request.app.state.repository
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving database is not ready",
        )
    return repository


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
def liveness() -> HealthResponse:
    """Proves the web process can receive and answer an HTTP request."""
    return HealthResponse(status="alive")


@app.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
def readiness(request: Request) -> ReadinessResponse:
    """Proves the model dependency loaded and predictions can be served."""
    bundle = require_bundle(request)
    require_repository(request).ping()
    return ReadinessResponse(status="ready", model_version=bundle.model_version)


@app.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
def model_information(request: Request) -> ModelInfoResponse:
    bundle = require_bundle(request)
    metadata = bundle.metadata
    return ModelInfoResponse(
        model_version=bundle.model_version,
        observation_window_days=metadata["observation_window_days"],
        prediction_horizon_days=metadata["prediction_horizon_days"],
        probability_threshold=bundle.probability_threshold,
        feature_count=len(bundle.feature_contract.all_features),
        training_snapshot_start=metadata["training_snapshot_start"],
        training_snapshot_end=metadata["training_snapshot_end"],
    )


@app.post("/v1/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Score one already-prepared customer feature record."""
    bundle = require_bundle(request)
    record = payload.features.model_dump()
    model_frame = pd.DataFrame([record], columns=bundle.feature_contract.all_features)
    probability = float(bundle.model.predict_proba(model_frame)[0, 1])
    return PredictionResponse(
        customer_id=payload.customer_id,
        snapshot_date=payload.snapshot_date,
        churn_probability=probability,
        above_model_threshold=probability >= bundle.probability_threshold,
        probability_threshold=bundle.probability_threshold,
        model_version=bundle.model_version,
    )


def decision_response(
    row: dict, strategy: str = "churn_prevention"
) -> CustomerDecisionResponse:
    if strategy == "churn_prevention":
        active_band = row["campaign_priority_band"]
        active_action = row["campaign_recommended_action"]
    elif strategy == "value_protection":
        active_band = row["value_protection_priority_band"]
        active_action = row["value_protection_recommended_action"]
    else:
        raise HTTPException(
            status_code=422,
            detail="strategy must be churn_prevention or value_protection",
        )
    return CustomerDecisionResponse(
        customer_id=row["customer_id"],
        snapshot_date=row["snapshot_date"],
        segment=row["segment"],
        customer_value=row["customer_value"],
        segment_frequency=row["segment_frequency"],
        churn_probability=row["churn_probability"],
        above_model_threshold=bool(row["above_model_threshold"]),
        probability_threshold=row["probability_threshold"],
        campaign_priority_band=row["campaign_priority_band"],
        campaign_recommended_action=row["campaign_recommended_action"],
        customer_value_cap=row["customer_value_cap"],
        value_at_risk_score=row["value_at_risk_score"],
        probability_rank=row["probability_rank"],
        value_at_risk_rank=row["value_at_risk_rank"],
        value_protection_priority_band=row["value_protection_priority_band"],
        value_protection_recommended_action=(
            row["value_protection_recommended_action"]
        ),
        active_strategy=strategy,
        active_priority_band=active_band,
        active_recommended_action=active_action,
        model_version=row["model_version"],
        scored_at=row["scored_at"].isoformat(),
    )


@app.get(
    "/v1/customers/{customer_id}",
    response_model=CustomerDecisionResponse,
    tags=["decisions"],
)
def get_customer_decision(
    customer_id: str,
    request: Request,
    strategy: str = "churn_prevention",
):
    row = require_repository(request).get_customer(customer_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return decision_response(row, strategy=strategy)


@app.get(
    "/v1/customers",
    response_model=list[CustomerDecisionResponse],
    tags=["decisions"],
)
def list_customer_decisions(
    request: Request,
    campaign_priority_band: str | None = None,
    value_protection_priority_band: str | None = None,
    segment: str | None = None,
    strategy: str = "churn_prevention",
    limit: int = 50,
):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    ordering = {
        "churn_prevention": "churn_probability",
        "value_protection": "value_at_risk",
    }
    if strategy not in ordering:
        raise HTTPException(
            status_code=422,
            detail="strategy must be churn_prevention or value_protection",
        )
    rows = require_repository(request).list_customers(
        campaign_priority_band=campaign_priority_band,
        value_protection_priority_band=value_protection_priority_band,
        segment=segment,
        order_by=ordering[strategy],
        limit=limit,
    )
    return [decision_response(row, strategy=strategy) for row in rows]
