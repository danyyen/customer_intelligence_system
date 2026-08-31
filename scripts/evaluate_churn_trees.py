"""Run the chronological Random Forest and gradient-boosting comparison.

Run from the repository root:
    python -m scripts.evaluate_churn_trees
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss

from customer_intelligence.churn import (
    baseline_metrics,
    build_tree_pipeline,
    final_temporal_holdout,
    make_logistic_feature_contract,
    purged_rolling_splits,
    rolling_tree_validation,
    tune_tree_probability_threshold,
)


CANDIDATES = {
    "RF — flexible": (
        "Random Forest",
        {
            "n_estimators": 150,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
    ),
    "RF — regularized": (
        "Random Forest",
        {
            "n_estimators": 150,
            "min_samples_leaf": 15,
            "max_features": "sqrt",
        },
    ),
    "HGB — regularized": (
        "Histogram Gradient Boosting",
        {
            "max_iter": 120,
            "learning_rate": 0.07,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 2.0,
        },
    ),
    "HGB — shallow": (
        "Histogram Gradient Boosting",
        {
            "max_iter": 120,
            "learning_rate": 0.07,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 50,
            "l2_regularization": 5.0,
        },
    ),
}


def main() -> None:
    data = pd.read_csv(
        Path("data/processed/churn_training_table.csv"),
        parse_dates=["SnapshotDate"],
    )
    contract = make_logistic_feature_contract(data)
    folds = purged_rolling_splits(
        data,
        pd.to_datetime(
            ["2011-02-01", "2011-03-01", "2011-04-01", "2011-05-01"]
        ),
        horizon_days=90,
    )
    selected, summary, validation_predictions = rolling_tree_validation(
        data, folds, CANDIDATES, contract
    )
    threshold, threshold_summary = tune_tree_probability_threshold(
        validation_predictions,
        selected,
        thresholds=[
            0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75,
        ],
    )

    development_idx, test_idx, _ = final_temporal_holdout(
        data, pd.Timestamp("2011-09-01"), horizon_days=90
    )
    model_name, parameters = CANDIDATES[selected]
    model = build_tree_pipeline(contract, model_name, parameters)
    model.fit(
        data.loc[development_idx, contract.all_features],
        data.loc[development_idx, "Churn"],
    )
    actual = data.loc[test_idx, "Churn"]
    probability = model.predict_proba(
        data.loc[test_idx, contract.all_features]
    )[:, 1]
    prediction = (probability >= threshold).astype("int8")
    metrics = {
        **baseline_metrics(actual, prediction, probability),
        "BrierScore": brier_score_loss(actual, probability),
        "PredictedHighRiskShare": prediction.mean(),
    }

    print("TREE VALIDATION SUMMARY")
    print(summary.round(4).to_string(index=False))
    print(f"\nSelected candidate: {selected}")
    print("\nTOP THRESHOLDS")
    print(threshold_summary.head(5).round(4).to_string(index=False))
    print(f"\nSelected threshold: {threshold:.2f}")
    print("\nUNTOUCHED SEPTEMBER TEST")
    print(pd.Series(metrics).round(4).to_string())


if __name__ == "__main__":
    main()
