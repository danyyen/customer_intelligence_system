"""Known-answer tests for leakage-safe historical feature engineering."""

import pandas as pd
import pytest

from customer_intelligence.churn import (
    build_asof_snapshot_features,
    build_churn_training_table,
)


SNAPSHOT = pd.Timestamp("2021-04-01")


def transaction(customer, stock, invoice, quantity, price, date, country="UK"):
    return {
        "Customer ID": customer,
        "StockCode": stock,
        "Invoice": invoice,
        "Quantity": quantity,
        "Price": price,
        "InvoiceDate": pd.Timestamp(date),
        "Country": country,
    }


def feature_transactions(include_future=True):
    rows = [
        # Customer 1 owns 10 units, then cancels 4 before the snapshot. Six
        # units at £2 must remain visible at the April snapshot.
        transaction(1, "A", "I1", 10, 2.0, "2021-01-15"),
        transaction(1, "A", "C1", -4, 2.0, "2021-03-01"),
        # Customer 2 has one known February order. A second order exactly on
        # the snapshot date belongs to the future and must not enter features.
        transaction(2, "B", "I2", 3, 5.0, "2021-02-01"),
        transaction(2, "B", "I3", 2, 5.0, "2021-04-01"),
        # Customer 3 is outside the 180-day eligibility window.
        transaction(3, "C", "I4", 1, 10.0, "2020-09-01"),
    ]
    if include_future:
        # This May cancellation must not rewrite April features.
        rows.append(transaction(1, "A", "C2", -6, 2.0, "2021-05-01"))
    return pd.DataFrame(rows)


def customer_row(features, customer_id):
    return features.loc[features["Customer ID"] == customer_id].iloc[0]


def test_fifo_state_uses_only_cancellations_observed_before_snapshot():
    features = build_asof_snapshot_features(
        feature_transactions(include_future=True),
        snapshot_dates=[SNAPSHOT],
    )
    customer = customer_row(features, 1)

    assert customer["LifetimeUnits"] == 6
    assert customer["LifetimeMonetary"] == pytest.approx(12.0)
    assert customer["Monetary90D"] == pytest.approx(12.0)
    assert customer["CancellationRows180D"] == 1
    assert customer["CancelledUnits180D"] == 4


def test_future_cancellation_cannot_change_earlier_features():
    with_future = build_asof_snapshot_features(
        feature_transactions(include_future=True),
        snapshot_dates=[SNAPSHOT],
    )
    without_future = build_asof_snapshot_features(
        feature_transactions(include_future=False),
        snapshot_dates=[SNAPSHOT],
    )

    columns = sorted(set(with_future.columns) - {"Customer ID"})
    pd.testing.assert_series_equal(
        customer_row(with_future, 1)[columns],
        customer_row(without_future, 1)[columns],
        check_names=False,
    )


def test_transaction_on_snapshot_is_not_a_historical_feature():
    customer = customer_row(
        build_asof_snapshot_features(
            feature_transactions(),
            snapshot_dates=[SNAPSHOT],
        ),
        2,
    )

    assert customer["LifetimeOrders"] == 1
    assert customer["LifetimeUnits"] == 3
    assert customer["LifetimeMonetary"] == pytest.approx(15.0)


def test_customer_outside_eligibility_window_is_excluded():
    features = build_asof_snapshot_features(
        feature_transactions(),
        snapshot_dates=[SNAPSHOT],
    )
    assert 3 not in set(features["Customer ID"])


def test_rolling_windows_and_trends_use_correct_time_ranges():
    model_df = pd.DataFrame(
        [
            # Days 91-180 before snapshot.
            transaction(1, "A", "OLD", 2, 10.0, "2020-12-01"),
            # Most recent 90 days.
            transaction(1, "B", "NEW", 3, 10.0, "2021-03-01"),
        ]
    )
    customer = customer_row(
        build_asof_snapshot_features(model_df, [SNAPSHOT]),
        1,
    )

    assert customer["Orders90D"] == 1
    assert customer["OrdersPrevious90D"] == 1
    assert customer["Orders180D"] == 2
    assert customer["Monetary90D"] == pytest.approx(30.0)
    assert customer["MonetaryPrevious90D"] == pytest.approx(20.0)
    assert customer["MonetaryChange90D"] == pytest.approx(10.0)


def test_training_table_keeps_target_but_drops_future_audit_fields():
    labels = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "SnapshotDate": [SNAPSHOT, SNAPSHOT],
            "Churn": [0, 1],
            "NextPurchaseDate": [pd.Timestamp("2021-04-10"), pd.NaT],
            "DaysToNextPurchase": [9.0, float("nan")],
            "ReturnedWithinHorizon": [1, 0],
        }
    )

    training = build_churn_training_table(
        feature_transactions(),
        labels,
    )

    assert len(training) == 2
    assert "Churn" in training.columns
    assert "NextPurchaseDate" not in training.columns
    assert "DaysToNextPurchase" not in training.columns
    assert "ReturnedWithinHorizon" not in training.columns


def test_missing_label_feature_record_fails_loudly():
    labels = pd.DataFrame(
        {
            "Customer ID": [999],
            "SnapshotDate": [SNAPSHOT],
            "Churn": [1],
        }
    )

    with pytest.raises(AssertionError, match="no as-of feature record"):
        build_churn_training_table(feature_transactions(), labels)
