"""Tests for leakage-safe churn tree models."""

import numpy as np
import pandas as pd
import pytest

from customer_intelligence.churn import (
    build_tree_pipeline,
    make_logistic_feature_contract,
    purged_rolling_splits,
    rolling_tree_validation,
    tune_tree_probability_threshold,
)


def tree_data():
    rows = []
    for month in pd.date_range("2020-10-01", "2021-03-01", freq="MS"):
        for customer, (recency, churn) in enumerate(
            [(10, 0), (30, 0), (90, 1), (150, 1)], start=1
        ):
            rows.append(
                {
                    "Customer ID": customer,
                    "SnapshotDate": month,
                    "Churn": churn,
                    "Recency": recency,
                    "Orders90D": np.nan if customer == 1 else 3 - churn,
                    "SnapshotMonth": month.month,
                    "SnapshotQuarter": month.quarter,
                    "IsHolidaySeason": int(month.month in (10, 11, 12)),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "model_name, parameters",
    [
        ("Random Forest", {"n_estimators": 20, "min_samples_leaf": 2}),
        ("Histogram Gradient Boosting", {"max_iter": 20, "min_samples_leaf": 2}),
    ],
)
def test_tree_pipelines_handle_missing_values_and_return_probabilities(
    model_name, parameters
):
    data = tree_data()
    contract = make_logistic_feature_contract(data)
    pipeline = build_tree_pipeline(contract, model_name, parameters)
    pipeline.fit(data[contract.all_features], data["Churn"])

    probabilities = pipeline.predict_proba(data[contract.all_features])[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_unknown_tree_model_is_rejected():
    contract = make_logistic_feature_contract(tree_data())
    with pytest.raises(ValueError, match="model_name"):
        build_tree_pipeline(contract, "Unknown model")


def test_rolling_tree_validation_returns_each_candidate():
    data = tree_data()
    folds = purged_rolling_splits(data, ["2021-03-01"], horizon_days=90)
    candidates = {
        "RF small": ("Random Forest", {"n_estimators": 20, "min_samples_leaf": 2}),
        "HGB small": ("Histogram Gradient Boosting", {"max_iter": 20, "min_samples_leaf": 2}),
    }
    selected, summary, predictions = rolling_tree_validation(
        data, folds, candidates
    )

    assert selected in candidates
    assert set(summary["Candidate"]) == set(candidates)
    assert len(predictions) == len(folds[0].validation_indices) * len(candidates)


def test_tree_threshold_is_selected_from_candidates():
    predictions = pd.DataFrame(
        {
            "Candidate": ["Tree"] * 6,
            "Actual": [0, 0, 0, 1, 1, 1],
            "Probability": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        }
    )
    selected, summary = tune_tree_probability_threshold(
        predictions, "Tree", [0.3, 0.5, 0.7]
    )
    assert selected == 0.5
    assert summary.loc[0, "BalancedAccuracy"] == 1
