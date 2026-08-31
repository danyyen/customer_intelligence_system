"""Known-answer tests for historical churn-label construction.

These tests protect the time boundary between historical model inputs and
future outcomes. A failure here would make every downstream model learn from
incorrect labels or information that was unavailable at prediction time.
"""

"""The tests protect the following rules.
1. Purchases before the snapshot are historical
2. A purchase on the snapshot date is future activity. 
That purchase must not become a model feature. It counts as a return during the prediction period
3. The observation boundary is correct
4. The prediction boundary is correct- 
The opening boundary is included and the ending boundary is excluded.
5. Incomplete future periods are rejected - If the dataset does not contain the complete following 90 days, 
the code raises an error instead of guessing the label.
6. Invalid settings are rejected
The tests prevent:
- zero-day observation windows;
- negative observation windows;
- zero-day horizons;
- negative horizons;
- missing required columns.

7. Labels are unique
There can be only one row for each: Customer ID + SnapshotDate + HorizonDays
This prevents accidental duplication in the training dataset.

"""


import pandas as pd
import pytest

from customer_intelligence.churn import (
    build_snapshot_labels,
    generate_monthly_snapshot_dates,
)


def purchase_day(customer_id, date, value=100.0, invoices=1):
    """Create one customer purchase-day record for a synthetic example."""
    return {
        "Customer ID": customer_id,
        "PurchaseDay": pd.Timestamp(date),
        "DailyValue": value,
        "Invoices": invoices,
    }


SNAPSHOT = pd.Timestamp("2021-04-01")


def boundary_data():
    """Synthetic history covering eligibility and label boundaries.

    Customer 1 returns exactly on the snapshot date.
    Customer 2 has eligible history but never returns.
    Customer 3 purchases exactly at the observation-window start.
    Customer 4 purchases one day before the observation window (ineligible).
    Customer 5 returns exactly at the exclusive 90-day label boundary.
    Customer 99 supplies sufficient dataset follow-up for complete labels.
    """
    return pd.DataFrame(
        [
            purchase_day(1, "2021-03-15", 50),
            purchase_day(1, "2021-04-01", 75),
            purchase_day(2, "2021-02-01", 60),
            purchase_day(3, "2021-01-01", 40),
            purchase_day(4, "2020-12-31", 30),
            purchase_day(5, "2021-03-01", 80),
            purchase_day(5, "2021-06-30", 90),
            purchase_day(99, "2021-03-20", 10),
            purchase_day(99, "2021-07-15", 10),
        ]
    )


def build_boundary_labels():
    return build_snapshot_labels(
        purchase_days=boundary_data(),
        snapshot_dates=[SNAPSHOT],
        horizons=[90],
        observation_days=90,
    )


def customer_row(labels, customer_id):
    return labels.loc[labels["Customer ID"] == customer_id].iloc[0]


def test_purchase_on_snapshot_is_future_return_not_history():
    labels = build_boundary_labels()
    customer = customer_row(labels, 1)

    assert customer["HistoryPurchaseDays"] == 1
    assert customer["LastPurchaseBeforeSnapshot"] == pd.Timestamp("2021-03-15")
    assert customer["NextPurchaseDate"] == SNAPSHOT
    assert customer["DaysToNextPurchase"] == 0
    assert customer["ReturnedWithinHorizon"] == 1
    assert customer["Churn"] == 0


def test_customer_with_no_future_purchase_is_churned():
    customer = customer_row(build_boundary_labels(), 2)

    assert pd.isna(customer["NextPurchaseDate"])
    assert customer["ReturnedWithinHorizon"] == 0
    assert customer["Churn"] == 1


def test_observation_start_is_inclusive_and_previous_day_is_excluded():
    labels = build_boundary_labels()

    assert 3 in set(labels["Customer ID"])
    assert 4 not in set(labels["Customer ID"])
    assert customer_row(labels, 3)["RecencyAtSnapshot"] == 90


def test_label_end_is_exclusive():
    customer = customer_row(build_boundary_labels(), 5)

    assert customer["LabelEndDate"] == pd.Timestamp("2021-06-30")
    assert customer["NextPurchaseDate"] == customer["LabelEndDate"]
    assert customer["ReturnedWithinHorizon"] == 0
    assert customer["Churn"] == 1


def test_multiple_horizons_create_unique_rows_and_longer_horizon_cannot_add_churn():
    labels = build_snapshot_labels(
        purchase_days=boundary_data(),
        snapshot_dates=[SNAPSHOT],
        horizons=[30, 60, 90],
        observation_days=90,
    )

    assert not labels[
        ["Customer ID", "SnapshotDate", "HorizonDays"]
    ].duplicated().any()

    churn_rates = labels.groupby("HorizonDays")["Churn"].mean()
    assert churn_rates.is_monotonic_decreasing


def test_incomplete_future_window_is_rejected():
    with pytest.raises(ValueError, match="complete future label window"):
        build_snapshot_labels(
            purchase_days=boundary_data(),
            snapshot_dates=[pd.Timestamp("2021-06-01")],
            horizons=[90],
            observation_days=90,
        )


def test_missing_required_column_is_rejected():
    bad_data = boundary_data().drop(columns="DailyValue")

    with pytest.raises(ValueError, match="missing required columns"):
        build_snapshot_labels(
            purchase_days=bad_data,
            snapshot_dates=[SNAPSHOT],
            horizons=[90],
            observation_days=90,
        )


@pytest.mark.parametrize("observation_days", [0, -1])
def test_nonpositive_observation_window_is_rejected(observation_days):
    with pytest.raises(ValueError, match="observation_days must be positive"):
        build_snapshot_labels(
            purchase_days=boundary_data(),
            snapshot_dates=[SNAPSHOT],
            horizons=[90],
            observation_days=observation_days,
        )


@pytest.mark.parametrize("horizon", [0, -30])
def test_nonpositive_horizon_is_rejected(horizon):
    with pytest.raises(ValueError, match="All horizons must be positive"):
        build_snapshot_labels(
            purchase_days=boundary_data(),
            snapshot_dates=[SNAPSHOT],
            horizons=[horizon],
            observation_days=90,
        )


def test_monthly_snapshot_dates_have_complete_history_and_future():
    purchase_days = pd.DataFrame(
        [
            purchase_day(1, "2021-01-01"),
            purchase_day(1, "2021-12-31"),
        ]
    )

    snapshots = generate_monthly_snapshot_dates(
        purchase_days=purchase_days,
        observation_days=90,
        horizon_days=90,
    )

    assert snapshots[0] == pd.Timestamp("2021-04-01")
    assert snapshots[-1] == pd.Timestamp("2021-10-01")
    assert all(snapshot.day == 1 for snapshot in snapshots)
