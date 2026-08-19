"""Cancellation-aware RFM (Recency, Frequency, Monetary) feature engineering.

The core problem this module solves: a customer who places a huge order and
cancels it minutes later must not be counted as a huge spender just because
the cancellation lives in a separate row. Two netting policies are provided:

- `net_group_fifo` (the primary, default policy): a cancellation offsets the
  OLDEST still-open purchase of the same item. Simple, doesn't depend on the
  Price field being reliable on every cancellation row.
- `net_group_price_compatible`: a cancellation only offsets a purchase at the
  SAME price, preferring the most recent match. More precise when price data
  is trustworthy, but on this dataset it changes results for ~9% of
  customers with negligible aggregate effect -- see the project README for
  the full comparison. Kept here as a validated alternative, not the default.

Both are validated against known-answer synthetic cases in tests/test_rfm.py.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


def net_group_fifo(group: pd.DataFrame) -> pd.Series:
    """FIFO netting within one (Customer, StockCode) group: a cancellation
    consumes the OLDEST still-open sale lot first, in chronological order.
    A cancellation can never reach into a lot that doesn't exist yet (a
    cancellation with nothing open to consume is an orphan -- no effect,
    it does not reach forward and erase a later, unrelated purchase).
    Returns remaining_quantity per row, indexed the same as `group`.
    """
    ordered = group.sort_values("InvoiceDate", kind="stable")
    remaining: dict = {}
    lots: list = []  # FIFO queue of [row_index, remaining_qty], oldest first

    for idx, qty in zip(ordered.index, ordered["Quantity"]):
        if qty > 0:
            lots.append([idx, qty])
            remaining[idx] = qty
        elif qty < 0:
            to_cancel = -qty
            for lot in lots:
                if to_cancel <= 0:
                    break
                consumed = min(lot[1], to_cancel)
                lot[1] -= consumed
                remaining[lot[0]] -= consumed
                to_cancel -= consumed
            remaining[idx] = 0
        else:
            remaining[idx] = 0

    return pd.Series(remaining)


def net_group_price_compatible(group: pd.DataFrame) -> pd.Series:
    """Price-compatible, closest-prior netting: a cancellation matches only
    against open sale lots at the SAME Price, preferring the most recently
    opened (closest-prior) matching lot first. Unmatched leftover (no open
    lot at a compatible price) is an orphan -- no effect, same floor-at-zero
    idea as FIFO, just price-aware and LIFO-within-price-match instead of
    strictly oldest-first.
    """
    ordered = group.sort_values("InvoiceDate", kind="stable")
    remaining: dict = {}
    lots: list = []  # chronological list of dicts: {idx, price, remaining}

    for idx, row in ordered.iterrows():
        qty = row["Quantity"]
        if qty > 0:
            lots.append({"idx": idx, "price": row["Price"], "remaining": qty})
            remaining[idx] = qty
        elif qty < 0:
            to_cancel = -qty
            cancel_price = row["Price"]
            candidates = [lot for lot in lots if lot["price"] == cancel_price and lot["remaining"] > 0]
            for lot in reversed(candidates):
                if to_cancel <= 0:
                    break
                consumed = min(lot["remaining"], to_cancel)
                lot["remaining"] -= consumed
                remaining[lot["idx"]] -= consumed
                to_cancel -= consumed
            remaining[idx] = 0
        else:
            remaining[idx] = 0

    return pd.Series(remaining)


def build_purchases_df(
    model_df: pd.DataFrame,
    net_func: Callable[[pd.DataFrame], pd.Series] = net_group_fifo,
) -> pd.DataFrame:
    """Apply a netting function per (Customer ID, StockCode) group and return
    only rows with remaining_quantity > 0, with a NetRevenue column attached
    (remaining_quantity x Price -- this, not raw Quantity x Price, is what
    Monetary should be built from).
    """
    remaining_quantity = (
        model_df.groupby(["Customer ID", "StockCode"], group_keys=False)
        .apply(net_func)
        .reindex(model_df.index)
    )

    purchases_df = model_df.copy()
    purchases_df["remaining_quantity"] = remaining_quantity
    purchases_df["NetRevenue"] = purchases_df["remaining_quantity"] * purchases_df["Price"]
    return purchases_df[purchases_df["remaining_quantity"] > 0].reset_index(drop=True)


def build_rfm_table(purchases_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """One row per Customer ID: Recency (days since their last net purchase,
    relative to snapshot_date), Frequency (count of distinct invoices with
    net-positive quantity), Monetary (sum of NetRevenue).

    Customers with zero net purchases (everything they touched netted to
    zero or less) are absent from `purchases_df` and therefore absent here
    -- by design, not an oversight; there's no purchase history to describe.
    """
    rfm_df = purchases_df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("NetRevenue", "sum"),
    ).reset_index()

    assert rfm_df.shape[0] == purchases_df["Customer ID"].nunique(), "one row per customer expected"
    assert rfm_df.isnull().sum().sum() == 0, "no nulls expected in the RFM table"
    assert (rfm_df["Monetary"] >= 0).all(), "Monetary should never be negative"

    return rfm_df
