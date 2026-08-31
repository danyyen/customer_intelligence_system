"""Chronological validation utilities for customer churn modelling.

Rows cannot be split randomly because one customer can appear at several
monthly snapshots and every label looks 90 days into the future. A training
label must finish before the later validation prediction date begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    """Row indices and dates for one expanding-window validation fold."""

    validation_date: pd.Timestamp
    train_indices: pd.Index
    validation_indices: pd.Index
    latest_train_snapshot: pd.Timestamp


def _prepared_dates(data: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column not in data.columns:
        raise ValueError(f"data must contain {date_column}")
    dates = pd.to_datetime(data[date_column]).dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{date_column} cannot contain missing values")
    return dates


def purged_rolling_splits(
    data: pd.DataFrame,
    validation_dates: Iterable[pd.Timestamp],
    horizon_days: int = 90,
    date_column: str = "SnapshotDate",
) -> list[TemporalFold]:
    """Create expanding chronological folds with a future-label safety gap.

    A row is eligible for training only when:

        training snapshot + prediction horizon <= validation snapshot

    This means the complete training outcome was already known when the
    validation prediction would have been made.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    dates = _prepared_dates(data, date_column)
    folds = []

    for raw_date in validation_dates:
        validation_date = pd.Timestamp(raw_date).normalize()
        validation_mask = dates.eq(validation_date)
        if not validation_mask.any():
            raise ValueError(
                f"No rows exist for validation date {validation_date.date()}"
            )

        # LabelEndDate is exclusive. If it equals the validation date, every
        # event used by that training label occurred before validation began.
        train_label_end = dates + pd.Timedelta(days=horizon_days)
        train_mask = train_label_end.le(validation_date)
        if not train_mask.any():
            raise ValueError(
                f"No safely labelled training rows exist before "
                f"{validation_date.date()}"
            )

        train_indices = data.index[train_mask]
        validation_indices = data.index[validation_mask]
        latest_train_snapshot = dates.loc[train_indices].max()

        assert (
            dates.loc[train_indices].max()
            + pd.Timedelta(days=horizon_days)
            <= validation_date
        )

        folds.append(
            TemporalFold(
                validation_date=validation_date,
                train_indices=train_indices,
                validation_indices=validation_indices,
                latest_train_snapshot=latest_train_snapshot,
            )
        )

    return folds


def final_temporal_holdout(
    data: pd.DataFrame,
    test_date: pd.Timestamp,
    horizon_days: int = 90,
    date_column: str = "SnapshotDate",
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Return development, untouched-test and purged row indices.

    ``development`` contains rows whose labels finish before the test snapshot.
    ``test`` contains only the selected final snapshot.
    ``purged`` contains intervening rows deliberately unused because their
    future label periods overlap the test prediction date.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    dates = _prepared_dates(data, date_column)
    test_date = pd.Timestamp(test_date).normalize()
    test_mask = dates.eq(test_date)
    if not test_mask.any():
        raise ValueError(f"No rows exist for test date {test_date.date()}")

    label_end = dates + pd.Timedelta(days=horizon_days)
    development_mask = label_end.le(test_date)
    purged_mask = ~(development_mask | test_mask)

    development_indices = data.index[development_mask]
    test_indices = data.index[test_mask]
    purged_indices = data.index[purged_mask]

    assert label_end.loc[development_indices].max() <= test_date
    return development_indices, test_indices, purged_indices
