"""Build the selected historical churn-label table.

The definition was selected in notebooks/02_churn_definition.ipynb:

* customer eligibility: at least one retained purchase in the prior 180 days;
* prediction schedule: month-start snapshots;
* churn outcome: no retained purchase in the following 90 days.

This script applies that final policy using every fully observable monthly
snapshot and writes the result used by the feature-engineering stage.
"""

from pathlib import Path

import pandas as pd

from customer_intelligence import build_model_df, load_transactions, tag_invoice_type
from customer_intelligence.churn import (
    build_snapshot_labels,
    generate_monthly_snapshot_dates,
)
from customer_intelligence.segmentation import build_purchases_df


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "uci data" / "online_retail_II.xlsx"
OUTPUT_PATH = ROOT / "data" / "processed" / "churn_labels_90d.csv"

OBSERVATION_DAYS = 180
HORIZON_DAYS = 90


def build_purchase_days() -> pd.DataFrame:
    """Create one cancellation-aware activity row per customer and day."""
    raw_df = load_transactions(DATA_PATH)
    tagged_df = tag_invoice_type(raw_df)
    model_df = build_model_df(tagged_df)
    purchases_df = build_purchases_df(model_df)

    order_df = (
        purchases_df
        .groupby(["Customer ID", "Invoice"], as_index=False)
        .agg(
            PurchaseDate=("InvoiceDate", "min"),
            OrderValue=("NetRevenue", "sum"),
        )
    )

    order_df["PurchaseDay"] = order_df["PurchaseDate"].dt.normalize()

    return (
        order_df
        .groupby(["Customer ID", "PurchaseDay"], as_index=False)
        .agg(
            DailyValue=("OrderValue", "sum"),
            Invoices=("Invoice", "nunique"),
        )
        .sort_values(["Customer ID", "PurchaseDay"])
        .reset_index(drop=True)
    )


def build_final_churn_labels() -> pd.DataFrame:
    """Apply the selected 180-day-history/90-day-outcome policy."""
    purchase_days = build_purchase_days()

    snapshot_dates = generate_monthly_snapshot_dates(
        purchase_days=purchase_days,
        observation_days=OBSERVATION_DAYS,
        horizon_days=HORIZON_DAYS,
    )

    labels = build_snapshot_labels(
        purchase_days=purchase_days,
        snapshot_dates=snapshot_dates,
        horizons=[HORIZON_DAYS],
        observation_days=OBSERVATION_DAYS,
    )

    # Keep future outcome details for audit, but downstream feature code must
    # use Churn as the target and must never use these future-only columns as
    # predictors: NextPurchaseDate, DaysToNextPurchase, LabelEndDate, or
    # ReturnedWithinHorizon.
    labels = labels.sort_values(
        ["SnapshotDate", "Customer ID"]
    ).reset_index(drop=True)

    assert labels["HorizonDays"].eq(HORIZON_DAYS).all()
    assert labels["Churn"].isin([0, 1]).all()
    assert not labels[
        ["Customer ID", "SnapshotDate", "HorizonDays"]
    ].duplicated().any()
    assert labels["LabelEndDate"].max() <= purchase_days["PurchaseDay"].max()

    return labels


def main() -> None:
    labels = build_final_churn_labels()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT_PATH, index=False)

    print(f"Built final churn labels: {len(labels):,} customer snapshots")
    print(f"Unique customers: {labels['Customer ID'].nunique():,}")
    print(f"Monthly snapshots: {labels['SnapshotDate'].nunique()}")
    print(
        "Snapshot period: "
        f"{labels['SnapshotDate'].min().date()} to "
        f"{labels['SnapshotDate'].max().date()}"
    )
    print(f"90-day churn rate: {labels['Churn'].mean():.1%}")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
