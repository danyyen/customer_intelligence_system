"""Typed HTTP request and response contracts.

Pydantic validates requests before model code runs. A malformed request gets a
clear 422 response instead of reaching scikit-learn and failing unpredictably.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class ChurnFeatures(BaseModel):
    """The exact 41-feature contract stored in the packaged model."""

    model_config = ConfigDict(extra="forbid")

    Recency: float = Field(ge=0)
    TenureDays: float = Field(ge=0)
    LifetimePurchaseDays: float = Field(ge=0)
    LifetimeOrders: float = Field(ge=0)
    LifetimeMonetary: float = Field(ge=0)
    LifetimeUnits: float = Field(ge=0)
    LifetimeUniqueProducts: float = Field(ge=0)
    AverageOrderValue: float = Field(ge=0)
    PurchaseDays30D: float = Field(ge=0)
    Orders30D: float = Field(ge=0)
    Monetary30D: float = Field(ge=0)
    Units30D: float = Field(ge=0)
    PurchaseDays60D: float = Field(ge=0)
    Orders60D: float = Field(ge=0)
    Monetary60D: float = Field(ge=0)
    Units60D: float = Field(ge=0)
    PurchaseDays90D: float = Field(ge=0)
    Orders90D: float = Field(ge=0)
    Monetary90D: float = Field(ge=0)
    Units90D: float = Field(ge=0)
    PurchaseDays180D: float = Field(ge=0)
    Orders180D: float = Field(ge=0)
    Monetary180D: float = Field(ge=0)
    Units180D: float = Field(ge=0)
    OrdersPrevious90D: float = Field(ge=0)
    MonetaryPrevious90D: float = Field(ge=0)
    PurchaseDaysPrevious90D: float = Field(ge=0)
    OrderChange90D: float
    MonetaryChange90D: float
    OrderTrendRatio90D: float = Field(ge=0)
    MonetaryTrendRatio90D: float = Field(ge=0)
    MedianPurchaseGap: float | None = Field(default=None, ge=0)
    LatestPurchaseGap: float | None = Field(default=None, ge=0)
    PurchaseGapStd: float | None = Field(default=None, ge=0)
    HasRepeatPurchaseHistory: int = Field(ge=0, le=1)
    CancellationRows180D: float = Field(ge=0)
    CancelledUnits180D: float = Field(ge=0)
    CancellationValue180D: float = Field(ge=0)
    PositiveUnits180D: float = Field(ge=0)
    CancellationUnitRate180D: float = Field(ge=0)
    SnapshotMonth: int = Field(ge=1, le=12)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: float
    snapshot_date: date
    features: ChurnFeatures


class PredictionResponse(BaseModel):
    customer_id: float
    snapshot_date: date
    churn_probability: float = Field(ge=0, le=1)
    above_model_threshold: bool
    probability_threshold: float = Field(ge=0, le=1)
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    observation_window_days: int
    prediction_horizon_days: int
    probability_threshold: float
    feature_count: int
    training_snapshot_start: str
    training_snapshot_end: str


class CustomerDecisionResponse(BaseModel):
    customer_id: str
    snapshot_date: date
    segment: str
    customer_value: float
    segment_frequency: float
    churn_probability: float = Field(ge=0, le=1)
    above_model_threshold: bool
    probability_threshold: float = Field(ge=0, le=1)
    campaign_priority_band: str
    campaign_recommended_action: str
    customer_value_cap: float
    value_at_risk_score: float
    probability_rank: int
    value_at_risk_rank: int
    value_protection_priority_band: str
    value_protection_recommended_action: str
    active_strategy: str
    active_priority_band: str
    active_recommended_action: str
    model_version: str
    scored_at: str
