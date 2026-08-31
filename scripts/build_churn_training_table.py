"""Build the leakage-safe customer-snapshot table for churn modelling.

Run from the repository root:

    python scripts/build_churn_training_table.py

The script joins two deliberately separate pieces:

1. ``churn_labels_90d.csv`` contains the future answer (Churn).
2. Historical transactions before each snapshot create the model features.

Keeping those pieces separate prevents the future answer from accidentally
entering the information supplied to the model.
"""

from pathlib import Path

import pandas as pd

from customer_intelligence import build_model_df, load_transactions, tag_invoice_type
from customer_intelligence.churn import build_churn_training_table


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "uci data" / "online_retail_II.xlsx"
LABEL_PATH = ROOT / "data" / "processed" / "churn_labels_90d.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "churn_training_table.csv"

# Multiple windows let the model distinguish immediate activity from longer
# behaviour. The largest window (180 days) is also the eligibility period.
FEATURE_WINDOWS = (30, 60, 90, 180)

# These columns describe the future and must never appear in the feature table.
FORBIDDEN_FUTURE_FEATURES = {
    "NextPurchaseDate",
    "DaysToNextPurchase",
    "LabelEndDate",
    "ReturnedWithinHorizon",
}


def main() -> None:
    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LABEL_PATH.relative_to(ROOT)}. Run "
            "python scripts/build_churn_labels.py first."
        )

    print("Loading and cleaning transaction history...")
    raw_df = load_transactions(DATA_PATH)
    tagged_df = tag_invoice_type(raw_df)
    model_df = build_model_df(tagged_df)

    print("Loading the selected 90-day churn labels...")
    labels = pd.read_csv(
        LABEL_PATH,
        parse_dates=["SnapshotDate"],
    )

    print("Building features using only transactions before each snapshot...")
    training = build_churn_training_table(
        model_df=model_df,
        labels=labels,
        windows=FEATURE_WINDOWS,
    )

    # Fail loudly if a future audit field accidentally enters the model table.
    leaked_columns = FORBIDDEN_FUTURE_FEATURES.intersection(training.columns)
    if leaked_columns:
        raise AssertionError(
            f"Future-only columns leaked into training data: {sorted(leaked_columns)}"
        )

    assert len(training) == len(labels), "Every label needs exactly one feature row"
    assert not training[["Customer ID", "SnapshotDate"]].duplicated().any()
    assert training["Churn"].isin([0, 1]).all()
    assert training["Recency"].between(1, max(FEATURE_WINDOWS)).all()
    assert (training["Monetary180D"] >= 0).all()
    assert (training["Orders180D"] >= 1).all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT_PATH, index=False)

    print(f"Built training rows: {len(training):,}")
    print(f"Features plus identifiers/target: {training.shape[1]:,} columns")
    print(f"Unique customers: {training['Customer ID'].nunique():,}")
    print(f"Snapshots: {training['SnapshotDate'].nunique()}")
    print(f"Churn rate: {training['Churn'].mean():.1%}")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
