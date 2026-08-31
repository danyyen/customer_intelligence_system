"""Leakage-safe tree-model utilities for churn prediction.

The models in this module use the same feature contract and chronological
folds as logistic regression. Preprocessing is fitted inside every fold, so a
validation month cannot influence missing-value fills or category handling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .baseline import baseline_metrics
from .modeling import LogisticFeatureContract, make_logistic_feature_contract
from .validation import TemporalFold


def build_tree_pipeline(
    contract: LogisticFeatureContract,
    model_name: str,
    parameters: Mapping[str, object] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a Random Forest or histogram-gradient-boosting pipeline."""
    parameters = dict(parameters or {})

    # Trees do not require scaling. We still impute inside the pipeline and
    # one-hot encode month because month 12 is not numerically 12 times month 1.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                list(contract.numeric_features),
            ),
            (
                "month",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                list(contract.categorical_features),
            ),
        ],
        remainder="drop",
    )

    if model_name == "Random Forest":
        defaults = {
            "n_estimators": 400,
            "n_jobs": -1,
            "random_state": random_state,
        }
        defaults.update(parameters)
        model = RandomForestClassifier(**defaults)
    elif model_name == "Histogram Gradient Boosting":
        model = HistGradientBoostingClassifier(
            random_state=random_state,
            **parameters,
        )
    else:
        raise ValueError(
            "model_name must be 'Random Forest' or "
            "'Histogram Gradient Boosting'"
        )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def rolling_tree_validation(
    data: pd.DataFrame,
    folds: Iterable[TemporalFold],
    candidates: Mapping[str, tuple[str, Mapping[str, object]]],
    contract: LogisticFeatureContract | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Compare tree candidates on purged chronological validation folds."""
    if not candidates:
        raise ValueError("candidates cannot be empty")
    contract = contract or make_logistic_feature_contract(data)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for fold_number, fold in enumerate(folds, start=1):
        train = data.loc[fold.train_indices]
        validation = data.loc[fold.validation_indices]

        for candidate_name, (model_name, parameters) in candidates.items():
            pipeline = build_tree_pipeline(
                contract, model_name=model_name, parameters=parameters
            )
            pipeline.fit(train[contract.all_features], train["Churn"])
            probabilities = pipeline.predict_proba(
                validation[contract.all_features]
            )[:, 1]
            predictions = (probabilities >= 0.5).astype("int8")
            metrics = baseline_metrics(
                validation["Churn"], predictions, probabilities
            )
            metric_rows.append(
                {
                    "Fold": fold_number,
                    "ValidationDate": fold.validation_date,
                    "Candidate": candidate_name,
                    **metrics,
                    "BrierScore": brier_score_loss(
                        validation["Churn"], probabilities
                    ),
                    "LogLoss": log_loss(
                        validation["Churn"], probabilities, labels=[0, 1]
                    ),
                }
            )
            prediction_rows.extend(
                {
                    "RowIndex": row_index,
                    "Fold": fold_number,
                    "ValidationDate": fold.validation_date,
                    "Candidate": candidate_name,
                    "Actual": int(actual),
                    "Probability": float(probability),
                }
                for row_index, actual, probability in zip(
                    validation.index, validation["Churn"], probabilities
                )
            )

    fold_results = pd.DataFrame(metric_rows)
    out_of_fold = pd.DataFrame(prediction_rows)
    summary = (
        fold_results.groupby("Candidate", as_index=False)
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
        .sort_values(
            ["MeanPRAUC", "MeanBrierScore", "MeanLiftAt20Pct"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )
    return str(summary.loc[0, "Candidate"]), summary, out_of_fold


def tune_tree_probability_threshold(
    out_of_fold: pd.DataFrame,
    selected_candidate: str,
    thresholds: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    """Choose a tree-model decision threshold using validation rows only."""
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("thresholds must be strictly between 0 and 1")
    selected = out_of_fold.loc[
        out_of_fold["Candidate"].eq(selected_candidate)
    ]
    if selected.empty:
        raise ValueError("No predictions exist for selected_candidate")

    rows = []
    for threshold in thresholds:
        predictions = (selected["Probability"] >= threshold).astype("int8")
        rows.append(
            {
                "ProbabilityThreshold": threshold,
                "PredictedHighRiskShare": predictions.mean(),
                **baseline_metrics(
                    selected["Actual"], predictions, selected["Probability"]
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["BalancedAccuracy", "F1"], ascending=[False, False]
    ).reset_index(drop=True)
    return float(summary.loc[0, "ProbabilityThreshold"]), summary
