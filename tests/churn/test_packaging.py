"""Tests for the serialized churn scoring contract."""

import pandas as pd
import pytest

from customer_intelligence.churn import (
    ChurnModelBundle,
    DEFAULT_ACTION_POLICY,
    build_logistic_pipeline,
    load_churn_bundle,
    make_logistic_feature_contract,
    save_churn_bundle,
)


def fitted_bundle():
    data = pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4],
            "SnapshotDate": pd.to_datetime(["2021-01-01"] * 4),
            "Churn": [0, 0, 1, 1],
            "Recency": [10, 30, 90, 150],
            "Orders90D": [4, 3, 1, 1],
            "SnapshotMonth": [1, 1, 1, 1],
            "SnapshotQuarter": [1, 1, 1, 1],
            "IsHolidaySeason": [0, 0, 0, 0],
        }
    )
    contract = make_logistic_feature_contract(data)
    model = build_logistic_pipeline(contract, regularization_c=0.01)
    model.fit(data[contract.all_features], data["Churn"])
    return data, ChurnModelBundle(
        model=model,
        feature_contract=contract,
        probability_threshold=0.4,
        model_version="test-1",
        metadata={"purpose": "unit test"},
        action_policy=DEFAULT_ACTION_POLICY,
    )


def test_bundle_scores_one_row_per_customer():
    data, bundle = fitted_bundle()
    scores = bundle.score(data)
    assert len(scores) == len(data)
    assert scores["ChurnProbability"].between(0, 1).all()
    assert scores["CampaignRecommendedAction"].notna().all()
    assert "PredictedChurn" not in scores
    assert "RiskBand" not in scores
    assert set(scores["ProbabilityThreshold"]) == {0.4}
    assert "AboveModelThreshold" in scores
    assert "CampaignPriorityBand" in scores
    assert set(scores["ModelVersion"]) == {"test-1"}


def test_bundle_rejects_missing_features():
    data, bundle = fitted_bundle()
    with pytest.raises(ValueError, match="Missing required"):
        bundle.score(data.drop(columns="Recency"))


def test_bundle_rejects_multiple_snapshots():
    data, bundle = fitted_bundle()
    data.loc[0, "SnapshotDate"] = pd.Timestamp("2021-02-01")
    with pytest.raises(ValueError, match="one snapshot"):
        bundle.score(data)


def test_bundle_round_trip_preserves_scores(tmp_path):
    data, bundle = fitted_bundle()
    path = save_churn_bundle(bundle, tmp_path / "bundle.joblib")
    restored = load_churn_bundle(path)
    pd.testing.assert_frame_equal(bundle.score(data), restored.score(data))
