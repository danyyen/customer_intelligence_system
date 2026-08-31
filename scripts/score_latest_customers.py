"""Build the latest eligible snapshot and create the decision-ready export.

Run after packaging the model:
    python -m scripts.score_latest_customers
"""

from pathlib import Path

import pandas as pd

from customer_intelligence import build_model_df, load_transactions, tag_invoice_type
from customer_intelligence.churn import (
    add_value_at_risk_ranking,
    VALUE_PROTECTION_ACTION_POLICY,
    build_asof_snapshot_features,
    load_churn_bundle,
)
from customer_intelligence.api.config import get_settings
from customer_intelligence.api.storage import DecisionRepository


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "uci data" / "online_retail_II.xlsx"
SEGMENT_PATH = ROOT / "data" / "processed" / "customer_segments.csv"
ARTIFACT_PATH = ROOT / "models" / "churn_logistic_v1.joblib"
OUTPUT_PATH = ROOT / "data" / "processed" / "latest_customer_decisions.csv"


def main() -> None:
    bundle = load_churn_bundle(ARTIFACT_PATH)
    transactions = build_model_df(tag_invoice_type(load_transactions(DATA_PATH)))
    snapshot_date = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    features = build_asof_snapshot_features(transactions, [snapshot_date])
    scores = bundle.score(features)

    segments = pd.read_csv(SEGMENT_PATH, parse_dates=["SnapshotDate"])
    segment_day = segments["SnapshotDate"].dt.normalize().unique()
    if len(segment_day) != 1 or pd.Timestamp(segment_day[0]) != snapshot_date:
        raise ValueError(
            "Segmentation and churn snapshots do not match. Rebuild segmentation "
            "before joining the decision table."
        )
    segment_context = segments.rename(
        columns={
            "SnapshotDate": "SegmentSnapshotDate",
            "Recency": "SegmentRecency",
            "Frequency": "SegmentFrequency",
            "Monetary": "CustomerValue",
        }
    )
    decisions = scores.merge(
        segment_context,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )
    if decisions["Segment"].isna().any():
        raise AssertionError("Every eligible scored customer must have a segment")
    decisions = decisions.sort_values("Customer ID").reset_index(drop=True)
    decisions = add_value_at_risk_ranking(decisions, cap_quantile=0.99)
    decisions["ValueProtectionRecommendedAction"] = (
        decisions["ValueProtectionPriorityBand"]
        .astype(str)
        .map(VALUE_PROTECTION_ACTION_POLICY)
    )

    columns = [
        "Customer ID", "SnapshotDate", "Segment", "CustomerValue",
        "SegmentFrequency", "ChurnProbability", "AboveModelThreshold",
        "ProbabilityThreshold", "CampaignPriorityBand", "CampaignRecommendedAction",
        "CustomerValueCap", "ValueAtRiskScore", "ProbabilityRank",
        "ValueAtRiskRank", "ValueProtectionPriorityBand",
        "ValueProtectionRecommendedAction", "ModelVersion",
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    decisions[columns].to_csv(OUTPUT_PATH, index=False)
    repository = DecisionRepository(get_settings().database_url)
    repository.initialize()
    run_id = repository.publish_snapshot(decisions[columns])
    print(f"Scored eligible customers: {len(decisions):,}")
    print("\nCampaign-priority counts:")
    print(decisions["CampaignPriorityBand"].value_counts(sort=False))
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Published database scoring run: {run_id}")


if __name__ == "__main__":
    main()
