"""Select and evaluate the leakage-safe logistic-regression churn model.

Run from the repository root:
    python -m scripts.evaluate_churn_logistic

Model settings are selected using historical validation months.  The final
September holdout is evaluated only after those settings have been frozen.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss

from customer_intelligence.churn.baseline import baseline_metrics, recency_rule
from customer_intelligence.churn.modeling import (
    build_logistic_pipeline,
    coefficient_table,
    make_logistic_feature_contract,
    rolling_logistic_validation,
    tune_probability_threshold,
)
from customer_intelligence.churn.validation import (
    final_temporal_holdout,
    purged_rolling_splits,
)


DATA_PATH = Path("data/processed/churn_training_table.csv")
VALIDATION_DATES = pd.to_datetime(
    ["2011-02-01", "2011-03-01", "2011-04-01", "2011-05-01"]
)
FINAL_TEST_DATE = pd.Timestamp("2011-09-01")
PREDICTION_HORIZON_DAYS = 90


def main() -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["SnapshotDate"])
    contract = make_logistic_feature_contract(data)

    # Compare regularization strengths using only historical validation folds.
    folds = purged_rolling_splits(
        data,
        validation_dates=VALIDATION_DATES,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )
    best_c, c_summary, validation_predictions = rolling_logistic_validation(
        data,
        folds,
        c_values=[0.001, 0.01, 0.1, 1.0, 10.0],
        contract=contract,
    )

    # The probability cutoff controls who is operationally called high risk.
    best_threshold, threshold_summary = tune_probability_threshold(
        validation_predictions,
        selected_c=best_c,
        thresholds=[
            0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75,
        ],
    )

    # Fit on all safely observable development rows, then open the holdout once.
    development_idx, test_idx, _ = final_temporal_holdout(
        data,
        test_date=FINAL_TEST_DATE,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )
    model = build_logistic_pipeline(contract, regularization_c=best_c)
    model.fit(
        data.loc[development_idx, contract.all_features],
        data.loc[development_idx, "Churn"],
    )

    y_test = data.loc[test_idx, "Churn"]
    probability = model.predict_proba(
        data.loc[test_idx, contract.all_features]
    )[:, 1]
    prediction = (probability >= best_threshold).astype(int)
    logistic_metrics = pd.Series(
        {
            **baseline_metrics(y_test, prediction, probability),
            "BrierScore": brier_score_loss(y_test, probability),
            "PredictedHighRiskShare": prediction.mean(),
        }
    )
    baseline_score = data.loc[test_idx, "Recency"].to_numpy()
    baseline_prediction = recency_rule(
        data.loc[test_idx, "Recency"], threshold_days=60
    )
    recency_metrics = pd.Series(
        baseline_metrics(y_test, baseline_prediction, baseline_score)
    )

    print("C SUMMARY")
    print(c_summary.round(4).to_string(index=False))
    print(f"\nSelected C: {best_c:g}")
    print("\nTOP PROBABILITY THRESHOLDS")
    print(
        threshold_summary.sort_values(
            ["BalancedAccuracy", "F1"], ascending=False
        ).head(5).round(4).to_string(index=False)
    )
    print(f"\nSelected probability threshold: {best_threshold:.2f}")
    print("\nFINAL TEST — LOGISTIC REGRESSION")
    print(logistic_metrics.round(4).to_string())
    print("\nFINAL TEST — 60-DAY RECENCY BASELINE")
    print(recency_metrics.round(4).to_string())
    print("\nMOST INFLUENTIAL STANDARDIZED COEFFICIENTS")
    print(
        coefficient_table(model)
        .sort_values("AbsoluteCoefficient", ascending=False)
        .head(12)
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
