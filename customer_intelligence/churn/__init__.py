"""Churn definition, feature engineering and modelling utilities."""

from .labels import (
    build_snapshot_labels,
    generate_monthly_snapshot_dates,
)
from .features import (
    build_asof_snapshot_features,
    build_churn_training_table,
)
from .validation import (
    TemporalFold,
    final_temporal_holdout,
    purged_rolling_splits,
)
from .baseline import (
    baseline_metrics,
    lift_at_fraction,
    recency_rule,
    tune_recency_threshold,
)

__all__ = [
    "build_snapshot_labels",
    "generate_monthly_snapshot_dates",
    "build_asof_snapshot_features",
    "build_churn_training_table",
    "TemporalFold",
    "final_temporal_holdout",
    "purged_rolling_splits",
    "baseline_metrics",
    "lift_at_fraction",
    "recency_rule",
    "tune_recency_threshold",
]
