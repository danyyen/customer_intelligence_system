"""Historical snapshot creation and churn-label engineering.

The functions in this module recreate what the business would have known
at historical prediction dates and then use a separate future period to
determine whether each customer returned.

Time convention:

    observation window     snapshot       prediction horizon
    ───────────────────────│───────────────────────────────
    historical features    prediction     future label

Purchases before the snapshot may be used as features. Purchases on or after
the snapshot may only be used to construct the future churn outcome.

The churn label is the answer the ML model will learn.
If the label is wrong, the model learns the wrong answer.
It may still produce impressive metrics, but those metrics would be unreliable.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_PURCHASE_DAY_COLUMNS = {
    "Customer ID",
    "PurchaseDay",
    "DailyValue",
    "Invoices",
}


def generate_monthly_snapshot_dates(
    purchase_days: pd.DataFrame,
    observation_days: int = 180,
    horizon_days: int = 90,
) -> pd.DatetimeIndex:
    """Generate valid month-start snapshot dates.

    The first snapshot must have a complete historical observation window.
    The final snapshot must have a completely observed future label window.
    """
    if observation_days <= 0:
        raise ValueError("observation_days must be positive")

    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    if "PurchaseDay" not in purchase_days.columns:
        raise ValueError("purchase_days must contain PurchaseDay")

    purchase_dates = pd.to_datetime(
        purchase_days["PurchaseDay"]
    ).dt.normalize()

    if purchase_dates.empty:
        raise ValueError("purchase_days cannot be empty")

    dataset_start = purchase_dates.min()
    dataset_end = purchase_dates.max()

    first_valid_snapshot = (
        dataset_start
        + pd.Timedelta(days=observation_days)
    )

    last_valid_snapshot = (
        dataset_end
        - pd.Timedelta(days=horizon_days)
    )

    if first_valid_snapshot > last_valid_snapshot:
        raise ValueError(
            "The dataset does not contain enough history for the "
            "requested observation and prediction windows."
        )

    return pd.date_range(
        start=first_valid_snapshot,
        end=last_valid_snapshot,
        freq="MS",
    )


def build_snapshot_labels(
    purchase_days: pd.DataFrame,
    snapshot_dates: Iterable[pd.Timestamp],
    horizons: Iterable[int] = (90,),
    observation_days: int = 180,
) -> pd.DataFrame:
    """Build customer-level churn labels at historical snapshots.

    Eligibility:
        A customer needs at least one retained purchase during
        [snapshot - observation_days, snapshot).

    Label:
        Churn = 0 when the customer purchases during
        [snapshot, snapshot + horizon).

        Churn = 1 when the customer makes no purchase during that period.

    The end of the prediction horizon is exclusive. For example, with a
    snapshot on 1 January and a 90-day label end on 1 April, purchases from
    1 January through 31 March count as returns.
    """
    missing_columns = REQUIRED_PURCHASE_DAY_COLUMNS.difference(
        purchase_days.columns
    )

    if missing_columns:
        raise ValueError(
            "purchase_days is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if observation_days <= 0:
        raise ValueError("observation_days must be positive")

    horizons = list(horizons)

    if not horizons:
        raise ValueError("At least one horizon is required")

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizons must be positive")

    prepared = purchase_days.copy()

    prepared["PurchaseDay"] = pd.to_datetime(
        prepared["PurchaseDay"]
    ).dt.normalize()

    prepared = prepared.sort_values(
        ["Customer ID", "PurchaseDay"]
    ).reset_index(drop=True)

    dataset_end = prepared["PurchaseDay"].max()
    output_frames = []

    for snapshot_date in pd.to_datetime(
        list(snapshot_dates)
    ):
        snapshot_date = pd.Timestamp(
            snapshot_date
        ).normalize()

        largest_label_end = (
            snapshot_date
            + pd.Timedelta(days=max(horizons))
        )

        if largest_label_end > dataset_end:
            raise ValueError(
                f"Snapshot {snapshot_date.date()} does not have "
                "a complete future label window."
            )

        observation_start = (
            snapshot_date
            - pd.Timedelta(days=observation_days)
        )

        history = prepared[
            (
                prepared["PurchaseDay"]
                >= observation_start
            )
            & (
                prepared["PurchaseDay"]
                < snapshot_date
            )
        ]

        if history.empty:
            continue

        eligible = (
            history
            .groupby("Customer ID", as_index=False)
            .agg(
                HistoryPurchaseDays=(
                    "PurchaseDay",
                    "nunique",
                ),
                HistoryInvoices=(
                    "Invoices",
                    "sum",
                ),
                HistoryMonetary=(
                    "DailyValue",
                    "sum",
                ),
                FirstPurchaseInWindow=(
                    "PurchaseDay",
                    "min",
                ),
                LastPurchaseBeforeSnapshot=(
                    "PurchaseDay",
                    "max",
                ),
            )
        )

        eligible["SnapshotDate"] = snapshot_date
        eligible["ObservationStart"] = observation_start

        eligible["RecencyAtSnapshot"] = (
            eligible["SnapshotDate"]
            - eligible["LastPurchaseBeforeSnapshot"]
        ).dt.days

        eligible["ObservedTenureInWindow"] = (
            eligible["LastPurchaseBeforeSnapshot"]
            - eligible["FirstPurchaseInWindow"]
        ).dt.days

        eligible["CustomerStageAtSnapshot"] = np.where(
            eligible["HistoryPurchaseDays"] == 1,
            "One purchase day",
            "Repeat purchase days",
        )

        future_activity = prepared[
            prepared["PurchaseDay"] >= snapshot_date
        ]

        next_purchase = (
            future_activity
            .groupby("Customer ID", as_index=False)
            .agg(
                NextPurchaseDate=(
                    "PurchaseDay",
                    "min",
                )
            )
        )

        eligible = eligible.merge(
            next_purchase,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )

        eligible["DaysToNextPurchase"] = (
            eligible["NextPurchaseDate"]
            - eligible["SnapshotDate"]
        ).dt.days

        for horizon in horizons:
            labelled = eligible.copy()

            labelled["HorizonDays"] = horizon

            labelled["LabelEndDate"] = (
                labelled["SnapshotDate"]
                + pd.Timedelta(days=horizon)
            )

            labelled["ReturnedWithinHorizon"] = (
                labelled["NextPurchaseDate"].notna()
                & (
                    labelled["NextPurchaseDate"]
                    < labelled["LabelEndDate"]
                )
            ).astype("int8")

            labelled["Churn"] = (
                1
                - labelled["ReturnedWithinHorizon"]
            ).astype("int8")

            output_frames.append(labelled)

    if not output_frames:
        raise ValueError(
            "No snapshot labels were created. Check the snapshot dates."
        )

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    if result[
        [
            "Customer ID",
            "SnapshotDate",
            "HorizonDays",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Customer-snapshot-horizon rows must be unique"
        )

    return result
