"""HTTP contract tests for the first FastAPI deployment layer."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from customer_intelligence.api.main import app
from customer_intelligence.api.storage import DecisionRepository
from customer_intelligence.churn import (
    ChurnModelBundle,
    DEFAULT_ACTION_POLICY,
    build_logistic_pipeline,
    make_logistic_feature_contract,
    save_churn_bundle,
)


def json_value(value):
    """Convert pandas/NumPy scalars to values a real JSON client can send."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = pd.read_csv(
        "data/processed/churn_training_table.csv",
        parse_dates=["SnapshotDate"],
    ).head(20)
    # Ensure the tiny test fit has both classes.
    source = pd.concat(
        [source[source["Churn"].eq(0)].head(5), source[source["Churn"].eq(1)].head(5)]
    )
    contract = make_logistic_feature_contract(source)
    model = build_logistic_pipeline(contract, regularization_c=0.01)
    model.fit(source[contract.all_features], source["Churn"])
    bundle = ChurnModelBundle(
        model=model,
        feature_contract=contract,
        probability_threshold=0.40,
        model_version="api-test-1",
        metadata={
            "observation_window_days": 180,
            "prediction_horizon_days": 90,
            "training_snapshot_start": "2010-06-01",
            "training_snapshot_end": "2011-09-01",
        },
        action_policy=DEFAULT_ACTION_POLICY.copy(),
    )
    artifact = save_churn_bundle(bundle, tmp_path / "test-model.joblib")
    monkeypatch.setenv("CHURN_MODEL_PATH", str(artifact))
    database_url = f"sqlite:///{tmp_path / 'api.db'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    repository = DecisionRepository(database_url)
    repository.initialize()
    repository.publish_snapshot(
        pd.DataFrame(
            {
                "Customer ID": ["100", "200"],
                "SnapshotDate": pd.to_datetime(["2021-04-01", "2021-04-01"]),
                "Segment": ["Champions", "Lapsed low-value"],
                "CustomerValue": [2000.0, 100.0],
                "SegmentFrequency": [10, 1],
                "ChurnProbability": [0.15, 0.90],
                "AboveModelThreshold": [0, 1],
                "ProbabilityThreshold": [0.4, 0.4],
                "CampaignPriorityBand": [
                    "Standard monitoring", "Priority 1 - Immediate"
                ],
                "CampaignRecommendedAction": ["Normal", "Call now"],
                "CustomerValueCap": [2000.0, 2000.0],
                "ValueAtRiskScore": [300.0, 90.0],
                "ProbabilityRank": [2, 1],
                "ValueAtRiskRank": [1, 2],
                "ValueProtectionPriorityBand": [
                    "Priority 1 - Immediate", "Standard monitoring"
                ],
                "ValueProtectionRecommendedAction": [
                    "Protect value now", "Monitor value"
                ],
                "ModelVersion": ["api-test-1", "api-test-1"],
            }
        )
    )
    with TestClient(app) as client:
        yield client, source.iloc[0], contract


def test_liveness_does_not_depend_on_model(api_client):
    client, _, _ = api_client
    assert client.get("/health/live").json() == {"status": "alive"}


def test_readiness_confirms_loaded_model(api_client):
    client, _, _ = api_client
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["model_version"] == "api-test-1"


def test_model_metadata_is_exposed(api_client):
    client, _, contract = api_client
    response = client.get("/v1/model")
    assert response.status_code == 200
    assert response.json()["feature_count"] == len(contract.all_features)


def test_valid_prediction_returns_probability_and_version(api_client):
    client, row, contract = api_client
    features = {
        name: json_value(row[name])
        for name in contract.all_features
    }
    response = client.post(
        "/v1/predict",
        json={
            "customer_id": json_value(row["Customer ID"]),
            "snapshot_date": str(row["SnapshotDate"].date()),
            "features": features,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_version"] == "api-test-1"
    assert "campaign_priority_band" not in body


def test_missing_feature_is_rejected_before_model(api_client):
    client, row, contract = api_client
    features = {
        name: json_value(row[name])
        for name in contract.all_features
        if name != "Recency"
    }
    response = client.post(
        "/v1/predict",
        json={
            "customer_id": json_value(row["Customer ID"]),
            "snapshot_date": str(row["SnapshotDate"].date()),
            "features": features,
        },
    )
    assert response.status_code == 422


def test_customer_lookup_returns_latest_published_decision(api_client):
    client, _, _ = api_client
    response = client.get("/v1/customers/200")
    assert response.status_code == 200
    assert response.json()["campaign_priority_band"] == "Priority 1 - Immediate"
    assert response.json()["model_version"] == "api-test-1"


def test_customer_list_filters_and_orders_by_probability(api_client):
    client, _, _ = api_client
    response = client.get("/v1/customers", params={"limit": 2})
    assert response.status_code == 200
    assert [row["customer_id"] for row in response.json()] == ["200", "100"]
    filtered = client.get(
        "/v1/customers",
        params={"campaign_priority_band": "Standard monitoring", "limit": 10},
    )
    assert [row["customer_id"] for row in filtered.json()] == ["100"]
    value_ordered = client.get(
        "/v1/customers", params={"strategy": "value_protection", "limit": 2}
    )
    assert [row["customer_id"] for row in value_ordered.json()] == ["100", "200"]
    assert value_ordered.json()[0]["active_strategy"] == "value_protection"
    assert value_ordered.json()[0]["active_recommended_action"] == "Protect value now"


def test_unknown_customer_returns_404(api_client):
    client, _, _ = api_client
    response = client.get("/v1/customers/does-not-exist")
    assert response.status_code == 404
