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
from .modeling import (
    LogisticFeatureContract,
    build_logistic_pipeline,
    coefficient_table,
    make_logistic_feature_contract,
    rolling_logistic_validation,
    tune_probability_threshold,
)
from .tree_modeling import (
    build_tree_pipeline,
    rolling_tree_validation,
    tune_tree_probability_threshold,
)
from .decisioning import (
    add_value_at_risk_ranking,
    assign_campaign_priority_bands,
    assign_risk_bands,
    capacity_table,
    learn_risk_band_cutoffs,
)
from .packaging import (
    ChurnModelBundle,
    DEFAULT_ACTION_POLICY,
    VALUE_PROTECTION_ACTION_POLICY,
    load_churn_bundle,
    save_churn_bundle,
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
    "LogisticFeatureContract",
    "build_logistic_pipeline",
    "coefficient_table",
    "make_logistic_feature_contract",
    "rolling_logistic_validation",
    "tune_probability_threshold",
    "build_tree_pipeline",
    "rolling_tree_validation",
    "tune_tree_probability_threshold",
    "assign_campaign_priority_bands",
    "add_value_at_risk_ranking",
    "assign_risk_bands",
    "capacity_table",
    "learn_risk_band_cutoffs",
    "ChurnModelBundle",
    "DEFAULT_ACTION_POLICY",
    "VALUE_PROTECTION_ACTION_POLICY",
    "load_churn_bundle",
    "save_churn_bundle",
]
