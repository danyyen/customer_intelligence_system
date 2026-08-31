"""Serializable churn model bundle and production-style scoring contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .decisioning import assign_campaign_priority_bands
from .modeling import LogisticFeatureContract


DEFAULT_ACTION_POLICY = {
    "Priority 1 - Immediate": (
        "Personalized outreach for the highest-ranked 10%; use segment and value"
    ),
    "Priority 2 - Targeted": "Targeted retention message for the next-ranked 10%",
    "Priority 3 - Nurture": "Automated email nurture or product reminder",
    "Standard monitoring": "Normal low-cost engagement; no paid retention incentive",
}

VALUE_PROTECTION_ACTION_POLICY = {
    "Priority 1 - Immediate": (
        "Account-manager outreach; prioritize the highest exposed customer value"
    ),
    "Priority 2 - Targeted": "Personalized retention offer for a high-value account",
    "Priority 3 - Nurture": "Low-cost relationship nurture for value protection",
    "Standard monitoring": "Monitor customer value and churn risk; no paid intervention",
}


@dataclass
class ChurnModelBundle:
    """Everything required to reproduce a churn score and business decision."""

    model: Pipeline
    feature_contract: LogisticFeatureContract
    probability_threshold: float
    model_version: str
    metadata: dict[str, Any]
    action_policy: dict[str, str]

    def validate_features(self, features: pd.DataFrame) -> None:
        required = {
            "Customer ID", "SnapshotDate", *self.feature_contract.all_features
        }
        missing = required.difference(features.columns)
        if missing:
            raise ValueError(f"Missing required scoring columns: {sorted(missing)}")
        if features.empty:
            raise ValueError("Scoring features cannot be empty")
        if features[["Customer ID", "SnapshotDate"]].duplicated().any():
            raise ValueError("Scoring rows must be unique per customer and snapshot")
        snapshot_days = pd.to_datetime(features["SnapshotDate"]).dt.normalize()
        if snapshot_days.nunique() != 1:
            raise ValueError("Score one snapshot at a time so capacity bands are valid")

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return probabilities, binary decisions, capacity bands and actions."""
        self.validate_features(features)
        ordered = features.sort_values("Customer ID").reset_index(drop=True)
        probability = self.model.predict_proba(
            ordered[self.feature_contract.all_features]
        )[:, 1]
        result = ordered[["Customer ID", "SnapshotDate"]].copy()
        result["ChurnProbability"] = probability
        result["AboveModelThreshold"] = (
            result["ChurnProbability"] >= self.probability_threshold
        ).astype("int8")
        result["ProbabilityThreshold"] = self.probability_threshold
        result["CampaignPriorityBand"] = assign_campaign_priority_bands(
            result["ChurnProbability"]
        )
        result["CampaignRecommendedAction"] = (
            result["CampaignPriorityBand"].astype(str).map(self.action_policy)
        )
        result["ModelVersion"] = self.model_version
        return result


def save_churn_bundle(bundle: ChurnModelBundle, path: str | Path) -> Path:
    """Persist a trusted local bundle. Joblib files must not be loaded untrusted."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_churn_bundle(path: str | Path) -> ChurnModelBundle:
    """Load a trusted bundle and verify its expected type."""
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ChurnModelBundle):
        raise TypeError("Artifact is not a ChurnModelBundle")
    return bundle
