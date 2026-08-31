"""Simple Recency-rule baseline and business-friendly evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .validation import TemporalFold


def recency_rule(recency: pd.Series, threshold_days: int) -> np.ndarray:
    """Predict churn when inactivity is at least ``threshold_days``."""
    if threshold_days <= 0:
        raise ValueError("threshold_days must be positive")
    if recency.isna().any():
        raise ValueError("Recency cannot contain missing values")
    return (recency.to_numpy() >= threshold_days).astype("int8")


def lift_at_fraction(
    y_true: Sequence[int],
    risk_score: Sequence[float],
    fraction: float = 0.20,
) -> float:
    """Churn rate in the highest-risk group divided by the overall rate."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    evaluation = pd.DataFrame({"Churn": y_true, "Risk": risk_score})
    if evaluation.empty:
        raise ValueError("evaluation data cannot be empty")

    selected_count = max(1, int(np.ceil(len(evaluation) * fraction)))
    selected = evaluation.nlargest(selected_count, "Risk")
    overall_rate = evaluation["Churn"].mean()
    if overall_rate == 0:
        return np.nan
    return selected["Churn"].mean() / overall_rate


def baseline_metrics(
    y_true: Sequence[int],
    predictions: Sequence[int],
    risk_score: Sequence[float],
) -> dict[str, float]:
    """Evaluate classification and ranking quality for a baseline/model."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    risk_score = np.asarray(risk_score)

    return {
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "BalancedAccuracy": balanced_accuracy_score(y_true, predictions),
        "ROCAUC": roc_auc_score(y_true, risk_score),
        "PRAUC": average_precision_score(y_true, risk_score),
        "LiftAt20Pct": lift_at_fraction(y_true, risk_score, fraction=0.20),
    }


def tune_recency_threshold(
    data: pd.DataFrame,
    folds: Iterable[TemporalFold],
    thresholds: Sequence[int],
    recency_column: str = "Recency",
    target_column: str = "Churn",
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Compare fixed Recency rules across later historical months.

    The threshold is selected by mean validation Balanced Accuracy. F1 is a
    secondary tie-breaker. The final test month must not be included here.
    """
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("thresholds must contain positive day counts")
    for column in (recency_column, target_column):
        if column not in data.columns:
            raise ValueError(f"data must contain {column}")

    fold_rows = []
    for fold_number, fold in enumerate(folds, start=1):
        validation = data.loc[fold.validation_indices]
        for threshold in thresholds:
            predictions = recency_rule(validation[recency_column], threshold)
            metrics = baseline_metrics(
                y_true=validation[target_column],
                predictions=predictions,
                risk_score=validation[recency_column],
            )
            fold_rows.append(
                {
                    "Fold": fold_number,
                    "ValidationDate": fold.validation_date,
                    "ThresholdDays": threshold,
                    **metrics,
                }
            )

    fold_results = pd.DataFrame(fold_rows)
    summary = (
        fold_results
        .groupby("ThresholdDays", as_index=False)
        .agg(
            MeanBalancedAccuracy=("BalancedAccuracy", "mean"),
            StdBalancedAccuracy=("BalancedAccuracy", "std"),
            MeanPrecision=("Precision", "mean"),
            MeanRecall=("Recall", "mean"),
            MeanF1=("F1", "mean"),
            MeanROCAUC=("ROCAUC", "mean"),
            MeanPRAUC=("PRAUC", "mean"),
            MeanLiftAt20Pct=("LiftAt20Pct", "mean"),
        )
    )
    ranked = summary.sort_values(
        ["MeanBalancedAccuracy", "MeanF1", "ThresholdDays"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_threshold = int(ranked.loc[0, "ThresholdDays"])
    return best_threshold, ranked, fold_results
