"""Leakage-safe logistic-regression utilities for churn modelling.

Every preprocessing step lives inside a scikit-learn Pipeline. This matters:
the median used to fill missing values and the mean/std used for scaling are
learned from each training fold only, never from validation or test months.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .baseline import baseline_metrics
from .validation import TemporalFold


IDENTIFIER_COLUMNS = {"Customer ID", "SnapshotDate"}
TARGET_COLUMN = "Churn"
FUTURE_ONLY_COLUMNS = {
    "NextPurchaseDate",
    "DaysToNextPurchase",
    "LabelEndDate",
    "ReturnedWithinHorizon",
}


@dataclass(frozen=True)
class LogisticFeatureContract:
    """The exact columns and roles accepted by the logistic model."""

    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


def make_logistic_feature_contract(data: pd.DataFrame) -> LogisticFeatureContract:
    """Select predictors while explicitly rejecting identifiers/future data."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"data must contain {TARGET_COLUMN}")

    leaked = FUTURE_ONLY_COLUMNS.intersection(data.columns)
    if leaked:
        raise ValueError(f"Future-only columns are forbidden: {sorted(leaked)}")

    # Month is categorical: December is not mathematically twelve times
    # January. Quarter and holiday flag repeat information already contained
    # in month, so they are excluded to avoid exact redundant calendar signals.
    categorical = ("SnapshotMonth",)
    redundant_calendar = {"SnapshotQuarter", "IsHolidaySeason"}
    excluded = IDENTIFIER_COLUMNS | {TARGET_COLUMN} | redundant_calendar

    missing_categorical = set(categorical).difference(data.columns)
    if missing_categorical:
        raise ValueError(
            f"Missing categorical features: {sorted(missing_categorical)}"
        )

    candidate_numeric = [
        column
        for column in data.columns
        if column not in excluded | set(categorical)
    ]
    non_numeric = [
        column
        for column in candidate_numeric
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise ValueError(
            "Unexpected non-numeric feature columns: "
            f"{sorted(non_numeric)}"
        )

    return LogisticFeatureContract(
        numeric_features=tuple(candidate_numeric),
        categorical_features=categorical,
    )


def build_logistic_pipeline(
    contract: LogisticFeatureContract,
    regularization_c: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Create median-imputation, scaling, one-hot and logistic steps."""
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")

    numeric_pipeline = Pipeline(
        steps=[
            # add_indicator=True tells the model which cadence values were
            # originally missing because the customer had only one purchase.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(contract.numeric_features)),
            (
                "month",
                categorical_pipeline,
                list(contract.categorical_features),
            ),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        C=regularization_c,
        solver="liblinear",
        max_iter=2_000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def rolling_logistic_validation(
    data: pd.DataFrame,
    folds: Iterable[TemporalFold],
    c_values: Sequence[float],
    contract: LogisticFeatureContract | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Select regularization strength using purged chronological folds.

    PR-AUC is the primary ranking metric. Brier score (lower is better) is the
    first tie-breaker because probabilities will later drive risk bands.
    """
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_values must contain positive values")
    contract = contract or make_logistic_feature_contract(data)

    metric_rows = []
    prediction_rows = []
    for fold_number, fold in enumerate(folds, start=1):
        train = data.loc[fold.train_indices]
        validation = data.loc[fold.validation_indices]

        for c_value in c_values:
            pipeline = build_logistic_pipeline(contract, c_value)
            pipeline.fit(train[contract.all_features], train[TARGET_COLUMN])
            probabilities = pipeline.predict_proba(
                validation[contract.all_features]
            )[:, 1]
            predictions = (probabilities >= 0.5).astype("int8")
            metrics = baseline_metrics(
                validation[TARGET_COLUMN], predictions, probabilities
            )
            metric_rows.append(
                {
                    "Fold": fold_number,
                    "ValidationDate": fold.validation_date,
                    "C": c_value,
                    **metrics,
                    "BrierScore": brier_score_loss(
                        validation[TARGET_COLUMN], probabilities
                    ),
                    "LogLoss": log_loss(
                        validation[TARGET_COLUMN], probabilities, labels=[0, 1]
                    ),
                }
            )
            for row_index, actual, probability in zip(
                validation.index,
                validation[TARGET_COLUMN],
                probabilities,
            ):
                prediction_rows.append(
                    {
                        "RowIndex": row_index,
                        "Fold": fold_number,
                        "ValidationDate": fold.validation_date,
                        "C": c_value,
                        "Actual": int(actual),
                        "Probability": float(probability),
                    }
                )

    fold_results = pd.DataFrame(metric_rows)
    out_of_fold = pd.DataFrame(prediction_rows)
    summary = (
        fold_results
        .groupby("C", as_index=False)
        .agg(
            MeanPRAUC=("PRAUC", "mean"),
            StdPRAUC=("PRAUC", "std"),
            MeanROCAUC=("ROCAUC", "mean"),
            MeanLiftAt20Pct=("LiftAt20Pct", "mean"),
            MeanBrierScore=("BrierScore", "mean"),
            MeanLogLoss=("LogLoss", "mean"),
            MeanBalancedAccuracy=("BalancedAccuracy", "mean"),
            MeanF1=("F1", "mean"),
        )
    )
    ranked = summary.sort_values(
        ["MeanPRAUC", "MeanBrierScore", "MeanLiftAt20Pct"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    best_c = float(ranked.loc[0, "C"])
    return best_c, ranked, out_of_fold


def tune_probability_threshold(
    out_of_fold: pd.DataFrame,
    selected_c: float,
    thresholds: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    """Choose a classification cutoff from validation probabilities only."""
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("thresholds must be strictly between 0 and 1")

    selected = out_of_fold[np.isclose(out_of_fold["C"], selected_c)].copy()
    if selected.empty:
        raise ValueError("No validation probabilities exist for selected_c")

    rows = []
    for threshold in thresholds:
        predictions = (selected["Probability"] >= threshold).astype("int8")
        metrics = baseline_metrics(
            selected["Actual"], predictions, selected["Probability"]
        )
        rows.append(
            {
                "ProbabilityThreshold": threshold,
                "PredictedHighRiskShare": predictions.mean(),
                **metrics,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["BalancedAccuracy", "F1"], ascending=[False, False]
    ).reset_index(drop=True)
    return float(summary.loc[0, "ProbabilityThreshold"]), summary


def coefficient_table(
    fitted_pipeline: Pipeline,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return standardized logistic coefficients with readable feature names."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    result = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coefficients}
    )
    result["AbsoluteCoefficient"] = result["Coefficient"].abs()
    result["Direction"] = np.where(
        result["Coefficient"] >= 0, "Higher churn risk", "Lower churn risk"
    )
    result = result.sort_values("AbsoluteCoefficient", ascending=False)
    if top_n is not None:
        result = result.head(top_n)
    return result.reset_index(drop=True)
