"""Tests for chronological splitting and the Recency-rule baseline."""

import numpy as np
import pandas as pd
import pytest

from customer_intelligence.churn import (
    TemporalFold,
    baseline_metrics,
    final_temporal_holdout,
    lift_at_fraction,
    purged_rolling_splits,
    recency_rule,
    tune_recency_threshold,
)


def monthly_rows():
    dates = pd.to_datetime(
        [
            "2020-10-01",
            "2020-11-01",
            "2020-12-01",
            "2021-01-01",
            "2021-02-01",
            "2021-03-01",
            "2021-06-01",
            "2021-07-01",
            "2021-08-01",
            "2021-09-01",
        ]
    )
    return pd.DataFrame(
        {
            "Customer ID": range(1, len(dates) + 1),
            "SnapshotDate": dates,
            "Recency": np.linspace(10, 100, len(dates)),
            "Churn": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_purged_fold_includes_only_labels_finished_before_validation():
    data = monthly_rows()
    fold = purged_rolling_splits(
        data,
        validation_dates=["2021-03-01"],
        horizon_days=90,
    )[0]

    train_dates = set(data.loc[fold.train_indices, "SnapshotDate"])
    # The 1 December label ends exactly on 1 March and is safe because the
    # label interval's ending boundary is exclusive.
    assert pd.Timestamp("2020-12-01") in train_dates
    assert pd.Timestamp("2021-01-01") not in train_dates
    assert set(data.loc[fold.validation_indices, "SnapshotDate"]) == {
        pd.Timestamp("2021-03-01")
    }


def test_final_holdout_keeps_test_untouched_and_identifies_purged_months():
    data = monthly_rows()
    development, test, purged = final_temporal_holdout(
        data,
        test_date="2021-09-01",
        horizon_days=90,
    )

    development_dates = set(data.loc[development, "SnapshotDate"])
    purged_dates = set(data.loc[purged, "SnapshotDate"])

    assert pd.Timestamp("2021-06-01") in development_dates
    assert pd.Timestamp("2021-07-01") in purged_dates
    assert pd.Timestamp("2021-08-01") in purged_dates
    assert set(data.loc[test, "SnapshotDate"]) == {pd.Timestamp("2021-09-01")}


def test_recency_rule_uses_inclusive_threshold():
    predictions = recency_rule(pd.Series([29, 30, 31]), threshold_days=30)
    assert predictions.tolist() == [0, 1, 1]


def test_lift_at_fraction_concentrates_high_risk_churners():
    y_true = [0, 0, 1, 1]
    risk = [0.1, 0.2, 0.8, 0.9]
    assert lift_at_fraction(y_true, risk, fraction=0.50) == pytest.approx(2.0)


def test_perfect_predictions_produce_perfect_core_metrics():
    metrics = baseline_metrics(
        y_true=[0, 0, 1, 1],
        predictions=[0, 0, 1, 1],
        risk_score=[0.1, 0.2, 0.8, 0.9],
    )
    assert metrics["Precision"] == 1
    assert metrics["Recall"] == 1
    assert metrics["F1"] == 1
    assert metrics["BalancedAccuracy"] == 1
    assert metrics["ROCAUC"] == 1
    assert metrics["PRAUC"] == 1


def test_threshold_tuning_selects_best_validation_rule():
    data = pd.DataFrame(
        {
            "SnapshotDate": pd.to_datetime(["2021-03-01"] * 4),
            "Recency": [10, 40, 80, 120],
            "Churn": [0, 0, 1, 1],
        }
    )
    fold = TemporalFold(
        validation_date=pd.Timestamp("2021-03-01"),
        train_indices=pd.Index([], dtype=int),
        validation_indices=data.index,
        latest_train_snapshot=pd.Timestamp("2020-12-01"),
    )

    best, summary, fold_results = tune_recency_threshold(
        data,
        folds=[fold],
        thresholds=[30, 60, 90],
    )

    assert best == 60
    assert summary.loc[0, "MeanBalancedAccuracy"] == 1
    assert len(fold_results) == 3
