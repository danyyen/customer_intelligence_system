"""Upgrade the latest decision export to the explicit priority contract.

This one-time-safe command reuses existing model probabilities; it does not
recompute features or predictions. Population ranks, the fixed threshold flag
and recommended actions are deterministic transformations of those scores.
"""

from pathlib import Path

import pandas as pd

from customer_intelligence.churn import (
    add_value_at_risk_ranking,
    VALUE_PROTECTION_ACTION_POLICY,
    assign_campaign_priority_bands,
    load_churn_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = ROOT / "data" / "processed" / "latest_customer_decisions.csv"
MODEL_PATH = ROOT / "models" / "churn_logistic_v1.joblib"


def main() -> None:
    decisions = pd.read_csv(DECISION_PATH, parse_dates=["SnapshotDate"])
    bundle = load_churn_bundle(MODEL_PATH)
    decisions = decisions.sort_values("Customer ID").reset_index(drop=True)

    # Recalculate instead of merely renaming so the file proves the documented
    # definitions and cannot preserve a stale value from an older contract.
    decisions["AboveModelThreshold"] = (
        decisions["ChurnProbability"] >= bundle.probability_threshold
    ).astype("int8")
    decisions["ProbabilityThreshold"] = bundle.probability_threshold
    decisions["CampaignPriorityBand"] = assign_campaign_priority_bands(
        decisions["ChurnProbability"]
    )
    decisions["CampaignRecommendedAction"] = (
        decisions["CampaignPriorityBand"].astype(str).map(bundle.action_policy)
    )
    decisions = add_value_at_risk_ranking(decisions, cap_quantile=0.99)
    decisions["ValueProtectionRecommendedAction"] = (
        decisions["ValueProtectionPriorityBand"]
        .astype(str)
        .map(VALUE_PROTECTION_ACTION_POLICY)
    )
    decisions = decisions.drop(
        columns=["PredictedChurn", "RiskBand", "RecommendedAction"], errors="ignore"
    )

    columns = [
        "Customer ID", "SnapshotDate", "Segment", "CustomerValue",
        "SegmentFrequency", "ChurnProbability", "AboveModelThreshold",
        "ProbabilityThreshold", "CampaignPriorityBand", "CampaignRecommendedAction",
        "CustomerValueCap", "ValueAtRiskScore", "ProbabilityRank",
        "ValueAtRiskRank", "ValueProtectionPriorityBand",
        "ValueProtectionRecommendedAction", "ModelVersion",
    ]
    decisions[columns].to_csv(DECISION_PATH, index=False)
    print(f"Refreshed {len(decisions):,} decisions")
    print(decisions["CampaignPriorityBand"].value_counts(sort=False))


if __name__ == "__main__":
    main()
