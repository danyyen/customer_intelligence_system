"""Leakage-safe customer features at historical prediction snapshots.

The central rule is deliberately simple:

    transactions before snapshot -> model features
    transactions after snapshot  -> never model features

The segmentation analysis reconciles cancellations over the complete dataset,
which is appropriate for a final descriptive view. Churn training is different:
we repeatedly pretend that a historical snapshot is "today". A cancellation
recorded next month must not rewrite what was known today.

To enforce that rule, this module processes transactions in chronological order.
At each snapshot it freezes the FIFO purchase-lot state created only from events
already observed, then calculates customer features from that frozen state.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (30, 60, 90, 180)

REQUIRED_TRANSACTION_COLUMNS = {
    "Customer ID",
    "StockCode",
    "Invoice",
    "Quantity",
    "Price",
    "InvoiceDate",
}


def _prepare_events(model_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize the cleaned transaction event table."""
    missing = REQUIRED_TRANSACTION_COLUMNS.difference(model_df.columns)
    if missing:
        raise ValueError(
            "model_df is missing required columns: "
            f"{sorted(missing)}"
        )

    columns = [
        "Customer ID",
        "StockCode",
        "Invoice",
        "Quantity",
        "Price",
        "InvoiceDate",
    ]
    if "Country" in model_df.columns:
        columns.append("Country")

    events = model_df[columns].copy()
    events["InvoiceDate"] = pd.to_datetime(events["InvoiceDate"])
    events["StockCode"] = events["StockCode"].astype(str)

    # Stable sorting preserves the original row order for events sharing the
    # exact timestamp, matching the documented segmentation FIFO policy.
    return events.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)


def _apply_event(open_lots, event) -> None:
    """Apply one observed sale/cancellation to the current FIFO state."""
    (
        customer_id,
        stock_code,
        invoice,
        quantity,
        price,
        invoice_date,
        country,
    ) = event
    key = (customer_id, stock_code)

    if quantity > 0:
        # A positive row opens a purchase lot. The mutable Remaining value is
        # reduced later only by cancellations that have already been observed.
        open_lots[key].append(
            {
                "Customer ID": customer_id,
                "StockCode": stock_code,
                "Invoice": invoice,
                "PurchaseDate": invoice_date,
                "Price": price,
                "Remaining": quantity,
                "Country": country,
            }
        )
        return

    if quantity >= 0:
        return

    to_cancel = -quantity
    queue = open_lots[key]

    # FIFO means the oldest still-open purchase quantity is consumed first.
    while to_cancel > 0 and queue:
        oldest_lot = queue[0]
        consumed = min(oldest_lot["Remaining"], to_cancel)
        oldest_lot["Remaining"] -= consumed
        to_cancel -= consumed

        if oldest_lot["Remaining"] == 0:
            queue.popleft()


def _open_lots_frame(open_lots) -> pd.DataFrame:
    """Convert the current in-memory FIFO state to a temporary dataframe."""
    records = [
        lot
        for queue in open_lots.values()
        for lot in queue
        if lot["Remaining"] > 0
    ]

    if not records:
        return pd.DataFrame(
            columns=[
                "Customer ID",
                "StockCode",
                "Invoice",
                "PurchaseDate",
                "Price",
                "Remaining",
                "Country",
            ]
        )

    lots = pd.DataFrame.from_records(records)
    lots["PurchaseDay"] = lots["PurchaseDate"].dt.normalize()
    lots["NetRevenue"] = lots["Remaining"] * lots["Price"]
    return lots


def _window_features(
    lots: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    windows: Sequence[int],
) -> pd.DataFrame:
    """Aggregate open purchase lots into one row per eligible customer."""
    lots = lots.copy()
    lots["DaysAgo"] = (snapshot_date - lots["PurchaseDay"]).dt.days

    # The largest window defines current eligibility. Older open purchases can
    # still support lifetime features, but cannot make an inactive customer
    # eligible for scoring.
    eligibility_window = max(windows)
    eligible_ids = lots.loc[
        lots["DaysAgo"].between(1, eligibility_window),
        "Customer ID",
    ].unique()

    if len(eligible_ids) == 0:
        return pd.DataFrame(columns=["Customer ID", "SnapshotDate"])

    eligible_lots = lots[lots["Customer ID"].isin(eligible_ids)].copy()

    lifetime = (
        eligible_lots
        .groupby("Customer ID", as_index=False)
        .agg(
            Recency=("PurchaseDay", lambda s: (snapshot_date - s.max()).days),
            TenureDays=("PurchaseDay", lambda s: (snapshot_date - s.min()).days),
            LifetimePurchaseDays=("PurchaseDay", "nunique"),
            LifetimeOrders=("Invoice", "nunique"),
            LifetimeMonetary=("NetRevenue", "sum"),
            LifetimeUnits=("Remaining", "sum"),
            LifetimeUniqueProducts=("StockCode", "nunique"),
        )
    )

    # Invoice values are calculated after reconciliation, so a partially
    # cancelled order contributes only its quantity still open at the snapshot.
    invoice_values = (
        eligible_lots
        .groupby(["Customer ID", "Invoice"], as_index=False)
        .agg(OrderValue=("NetRevenue", "sum"))
    )
    average_order_value = (
        invoice_values
        .groupby("Customer ID", as_index=False)
        .agg(AverageOrderValue=("OrderValue", "mean"))
    )
    features = lifetime.merge(
        average_order_value,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )

    for days in windows:
        window_lots = eligible_lots[
            eligible_lots["DaysAgo"].between(1, days)
        ]
        summary = (
            window_lots
            .groupby("Customer ID", as_index=False)
            .agg(**{
                f"PurchaseDays{days}D": ("PurchaseDay", "nunique"),
                f"Orders{days}D": ("Invoice", "nunique"),
                f"Monetary{days}D": ("NetRevenue", "sum"),
                f"Units{days}D": ("Remaining", "sum"),
            })
        )
        features = features.merge(
            summary,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )

    rolling_columns = [
        column
        for column in features.columns
        if any(
            column.startswith(prefix)
            for prefix in ("PurchaseDays", "Orders", "Monetary", "Units")
        )
        and column[-1] == "D"
    ]
    features[rolling_columns] = features[rolling_columns].fillna(0)

    # Compare the most recent 90 days with days 91-180. These features tell the
    # model whether engagement is rising, stable, or falling.
    previous_90 = eligible_lots[
        eligible_lots["DaysAgo"].between(91, 180)
    ]
    previous_summary = (
        previous_90
        .groupby("Customer ID", as_index=False)
        .agg(
            OrdersPrevious90D=("Invoice", "nunique"),
            MonetaryPrevious90D=("NetRevenue", "sum"),
            PurchaseDaysPrevious90D=("PurchaseDay", "nunique"),
        )
    )
    features = features.merge(
        previous_summary,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )
    previous_columns = [
        "OrdersPrevious90D",
        "MonetaryPrevious90D",
        "PurchaseDaysPrevious90D",
    ]
    features[previous_columns] = features[previous_columns].fillna(0)

    features["OrderChange90D"] = (
        features["Orders90D"] - features["OrdersPrevious90D"]
    )
    features["MonetaryChange90D"] = (
        features["Monetary90D"] - features["MonetaryPrevious90D"]
    )
    features["OrderTrendRatio90D"] = (
        (features["Orders90D"] + 1)
        / (features["OrdersPrevious90D"] + 1)
    )
    features["MonetaryTrendRatio90D"] = (
        (features["Monetary90D"] + 1)
        / (features["MonetaryPrevious90D"] + 1)
    )

    # Purchase cadence uses only completed gaps visible before the snapshot.
    cadence_records = []
    for customer_id, customer_lots in eligible_lots.groupby("Customer ID"):
        days = np.sort(customer_lots["PurchaseDay"].drop_duplicates().to_numpy())
        gaps = np.diff(days).astype("timedelta64[D]").astype(float)
        cadence_records.append(
            {
                "Customer ID": customer_id,
                "MedianPurchaseGap": np.median(gaps) if len(gaps) else np.nan,
                "LatestPurchaseGap": gaps[-1] if len(gaps) else np.nan,
                "PurchaseGapStd": np.std(gaps, ddof=0) if len(gaps) else np.nan,
            }
        )
    cadence = pd.DataFrame(cadence_records)
    features = features.merge(
        cadence,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )
    features["HasRepeatPurchaseHistory"] = (
        features["LifetimePurchaseDays"] >= 2
    ).astype("int8")

    features["SnapshotDate"] = snapshot_date
    features["SnapshotMonth"] = snapshot_date.month
    features["SnapshotQuarter"] = snapshot_date.quarter
    features["IsHolidaySeason"] = int(snapshot_date.month in (10, 11, 12))
    return features


def _cancellation_features(
    events_seen: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    observation_days: int,
) -> pd.DataFrame:
    """Summarize cancellation signals observed before the snapshot."""
    start = snapshot_date - pd.Timedelta(days=observation_days)
    recent = events_seen[
        (events_seen["InvoiceDate"] >= start)
        & (events_seen["InvoiceDate"] < snapshot_date)
    ].copy()

    if recent.empty:
        return pd.DataFrame(columns=["Customer ID"])

    recent["PositiveUnits"] = recent["Quantity"].clip(lower=0)
    recent["CancelledUnits"] = (-recent["Quantity"]).clip(lower=0)
    recent["CancellationValue"] = recent["CancelledUnits"] * recent["Price"]
    recent["IsCancellation"] = recent["Quantity"] < 0

    summary = (
        recent
        .groupby("Customer ID", as_index=False)
        .agg(
            CancellationRows180D=("IsCancellation", "sum"),
            CancelledUnits180D=("CancelledUnits", "sum"),
            CancellationValue180D=("CancellationValue", "sum"),
            PositiveUnits180D=("PositiveUnits", "sum"),
        )
    )
    summary["CancellationUnitRate180D"] = (
        summary["CancelledUnits180D"]
        / summary["PositiveUnits180D"].replace(0, np.nan)
    ).fillna(0)
    return summary


def build_asof_snapshot_features(
    model_df: pd.DataFrame,
    snapshot_dates: Iterable[pd.Timestamp],
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Create leakage-safe customer features for historical snapshots.

    Parameters
    ----------
    model_df:
        Cleaned customer-attributed merchandise rows, including sales and
        cancellations. It should be produced by ``build_model_df``.
    snapshot_dates:
        Historical prediction dates. Only earlier transactions are processed
        before each snapshot is frozen.
    windows:
        Rolling lookback windows. The largest window defines eligibility.
    """
    if not windows or any(days <= 0 for days in windows):
        raise ValueError("windows must contain positive day counts")
    if 90 not in windows or 180 not in windows:
        raise ValueError("windows must include 90 and 180 days for trend features")

    events = _prepare_events(model_df).rename(
        columns={
            "Customer ID": "CustomerID",
        }
    )
    if "Country" not in events.columns:
        events["Country"] = "Unknown"

    snapshots = sorted({pd.Timestamp(date).normalize() for date in snapshot_dates})
    if not snapshots:
        raise ValueError("At least one snapshot date is required")

    open_lots = defaultdict(deque)
    event_position = 0
    event_iterator = iter(events.itertuples(index=False, name=None))
    current_event = next(event_iterator, None)
    outputs = []

    for snapshot_date in snapshots:
        # Process only events strictly before the snapshot. This one comparison
        # is the main protection against future-data leakage.
        while (
            current_event is not None
            and current_event[5] < snapshot_date
        ):
            _apply_event(open_lots, current_event)
            event_position += 1
            current_event = next(event_iterator, None)

        lots = _open_lots_frame(open_lots)
        if lots.empty:
            continue

        snapshot_features = _window_features(lots, snapshot_date, windows)
        if snapshot_features.empty:
            continue

        events_seen = events.iloc[:event_position].rename(
            columns={"CustomerID": "Customer ID"}
        )
        cancellation = _cancellation_features(
            events_seen,
            snapshot_date,
            max(windows),
        )
        snapshot_features = snapshot_features.merge(
            cancellation,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
        cancellation_columns = [
            "CancellationRows180D",
            "CancelledUnits180D",
            "CancellationValue180D",
            "PositiveUnits180D",
            "CancellationUnitRate180D",
        ]
        snapshot_features[cancellation_columns] = (
            snapshot_features[cancellation_columns].fillna(0)
        )
        outputs.append(snapshot_features)

    if not outputs:
        raise ValueError("No snapshot features were created")

    result = pd.concat(outputs, ignore_index=True)
    if result[["Customer ID", "SnapshotDate"]].duplicated().any():
        raise AssertionError("Customer-snapshot feature rows must be unique")
    return result


def build_churn_training_table(
    model_df: pd.DataFrame,
    labels: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Join leakage-safe features to the selected historical churn target."""
    required_labels = {"Customer ID", "SnapshotDate", "Churn"}
    missing = required_labels.difference(labels.columns)
    if missing:
        raise ValueError(f"labels is missing required columns: {sorted(missing)}")

    label_keys = labels[["Customer ID", "SnapshotDate", "Churn"]].copy()
    label_keys["SnapshotDate"] = pd.to_datetime(label_keys["SnapshotDate"])
    if label_keys[["Customer ID", "SnapshotDate"]].duplicated().any():
        raise ValueError("labels must contain one row per customer and snapshot")

    features = build_asof_snapshot_features(
        model_df=model_df,
        snapshot_dates=label_keys["SnapshotDate"].unique(),
        windows=windows,
    )
    training = label_keys.merge(
        features,
        on=["Customer ID", "SnapshotDate"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_features = training["_merge"].ne("both")
    if missing_features.any():
        examples = training.loc[
            missing_features, ["Customer ID", "SnapshotDate"]
        ].head().to_dict("records")
        raise AssertionError(
            "Some labelled rows have no as-of feature record. Examples: "
            f"{examples}"
        )

    training = training.drop(columns="_merge")
    return training.sort_values(
        ["SnapshotDate", "Customer ID"]
    ).reset_index(drop=True)
