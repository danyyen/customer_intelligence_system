"""Build the Stage 1 data contract: a clean, customer-level segmentation table.

Runs the full pipeline (load -> clean -> RFM -> hybrid segmentation) using
the customer_intelligence package and writes one row per customer to
data/processed/customer_segments.csv. This is the interface later stages
(churn prediction, the eventual dashboard) should read from, instead of
re-running the notebook -- the notebook stays the analysis narrative, this
script is the reproducible pipeline that produces its output on demand.
"""

from pathlib import Path

import pandas as pd

from customer_intelligence import build_model_df, load_transactions, tag_invoice_type
from customer_intelligence.segmentation import (
    build_purchases_df,
    build_rfm_table,
    hybrid_segments,
    scale_rfm,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "uci data" / "online_retail_II.xlsx"
OUTPUT_PATH = ROOT / "data" / "processed" / "customer_segments.csv"

# Human-readable segment names, matching the notebook's analysis. K-Means is
# seeded (random_state=42) and the pipeline is deterministic, so this cluster
# numbering has been stable every time it's been run -- but EXPECTED_SEGMENT_SIZES
# below is a guard against a silent mislabeling if that ever stops being true
# (a different scikit-learn version, different input data), rather than
# trusting the mapping blindly.
SEGMENT_NAMES = {
    "0": "Recent developing",
    "1": "Champions",
    "2": "Lapsed low-value",
    "3": "At-risk established",
    "Outlier": "Exceptional high-value",
}

EXPECTED_SEGMENT_SIZES = {
    "Recent developing": 1208,
    "Champions": 1186,
    "Lapsed low-value": 1937,
    "At-risk established": 1427,
    "Exceptional high-value": 82,
}


def build_segmentation_table() -> pd.DataFrame:
    df = tag_invoice_type(load_transactions(DATA_PATH))
    model_df = build_model_df(df)
    purchases_df = build_purchases_df(model_df)

    snapshot_date = model_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm_df = build_rfm_table(purchases_df, snapshot_date)

    rfm_scaled, _ = scale_rfm(rfm_df)
    hybrid_df, _, _ = hybrid_segments(rfm_df, rfm_scaled)

    hybrid_df = hybrid_df.copy()
    hybrid_df["Segment"] = hybrid_df["Segment"].map(SEGMENT_NAMES)
    hybrid_df["SnapshotDate"] = snapshot_date

    return hybrid_df[["Customer ID", "Segment", "Recency", "Frequency", "Monetary", "SnapshotDate"]]


def main() -> None:
    table = build_segmentation_table()

    print(f"Built segmentation table: {len(table):,} customers")
    print("\nSegment sizes:")
    sizes = table["Segment"].value_counts()
    print(sizes)

    mismatches = {
        name: (sizes.get(name, 0), expected)
        for name, expected in EXPECTED_SEGMENT_SIZES.items()
        if sizes.get(name, 0) != expected
    }
    if mismatches:
        print("\nWARNING: segment sizes differ from the last verified run -- the")
        print("SEGMENT_NAMES mapping may no longer match cluster numbering. Check")
        print("before trusting the 'Segment' labels below:")
        for name, (actual, expected) in mismatches.items():
            print(f"  {name}: got {actual}, expected {expected}")
    else:
        print("\nSegment sizes match the last verified run -- labels are trustworthy.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
