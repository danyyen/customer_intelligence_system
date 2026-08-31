"""Tests for leakage-safe logistic-regression preprocessing and validation."""

import numpy as np
import pandas as pd
import pytest

from customer_intelligence.churn import (
    build_logistic_pipeline,
    coefficient_table,
    make_logistic_feature_contract,
    purged_rolling_splits,
    rolling_logistic_validation,
    tune_probability_threshold,
)


def modelling_data():
    rows = []
    row_id = 1
    for month in pd.date_range("2020-10-01", "2021-03-01", freq="MS"):
        for recency, churn in [(15, 0), (35, 0), (80, 1), (130, 1)]:
            rows.append(
                {
                    "Customer ID": row_id,
                    "SnapshotDate": month,
                    "Churn": churn,
                    "Recency": recency,
                    "Monetary90D": np.nan if row_id % 5 == 0 else 500 - recency,
                    "Orders90D": 3 if churn == 0 else 1,
                    "SnapshotMonth": month.month,
                    "SnapshotQuarter": month.quarter,
                    "IsHolidaySeason": int(month.month in (10, 11, 12)),
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def test_feature_contract_excludes_identifiers_target_and_redundant_calendar():
    contract = make_logistic_feature_contract(modelling_data())

    assert "Customer ID" not in contract.all_features
    assert "SnapshotDate" not in contract.all_features
    assert "Churn" not in contract.all_features
    assert "SnapshotQuarter" not in contract.all_features
    assert "IsHolidaySeason" not in contract.all_features
    assert contract.categorical_features == ("SnapshotMonth",)


def test_future_only_column_is_rejected():
    data = modelling_data()
    data["DaysToNextPurchase"] = 10

    with pytest.raises(ValueError, match="Future-only columns"):
        make_logistic_feature_contract(data)


def test_pipeline_imputes_scales_and_predicts_probabilities():
    data = modelling_data()
    contract = make_logistic_feature_contract(data)
    pipeline = build_logistic_pipeline(contract, regularization_c=0.1)
    pipeline.fit(data[contract.all_features], data["Churn"])

    probabilities = pipeline.predict_proba(data[contract.all_features])[:, 1]
    assert len(probabilities) == len(data)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_rolling_validation_returns_one_probability_per_validation_row_and_c():
    data = modelling_data()
    folds = purged_rolling_splits(
        data,
        validation_dates=["2021-03-01"],
        horizon_days=90,
    )
    c_values = [0.01, 0.1, 1.0]

    best_c, summary, out_of_fold = rolling_logistic_validation(
        data,
        folds=folds,
        c_values=c_values,
    )

    assert best_c in c_values
    assert set(summary["C"]) == set(c_values)
    assert len(out_of_fold) == len(folds[0].validation_indices) * len(c_values)
    assert out_of_fold["Probability"].between(0, 1).all()


def test_probability_threshold_is_selected_from_requested_candidates():
    out_of_fold = pd.DataFrame(
        {
            "C": [0.1] * 6,
            "Actual": [0, 0, 0, 1, 1, 1],
            "Probability": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        }
    )
    best, summary = tune_probability_threshold(
        out_of_fold,
        selected_c=0.1,
        thresholds=[0.3, 0.5, 0.7],
    )

    assert best == 0.5
    assert summary.loc[0, "BalancedAccuracy"] == 1


def test_coefficient_table_has_one_row_per_transformed_feature():
    data = modelling_data()
    contract = make_logistic_feature_contract(data)
    pipeline = build_logistic_pipeline(contract)
    pipeline.fit(data[contract.all_features], data["Churn"])

    coefficients = coefficient_table(pipeline)
    assert not coefficients.empty
    assert set(coefficients["Direction"]) <= {
        "Higher churn risk",
        "Lower churn risk",
    }
    assert coefficients["AbsoluteCoefficient"].is_monotonic_decreasing
