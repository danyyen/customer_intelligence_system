"""Known-answer tests for the cancellation-netting logic.

These are the same 7 hand-constructed cases used to validate the netting
functions during development (see the project notebook's "QA Check" section
for the real-data regression checks that accompanied them) -- turned into
real, independently re-runnable tests rather than one-off notebook cells.
Each case has an expected answer worked out by hand, not derived from the
code under test.
"""

import pandas as pd
import pytest

from customer_intelligence.segmentation.rfm import net_group_fifo, net_group_price_compatible


def make_row(customer_id, stockcode, invoice, qty, price, date):
    return {
        "Customer ID": customer_id,
        "StockCode": stockcode,
        "Invoice": invoice,
        "Quantity": qty,
        "Price": price,
        "InvoiceDate": pd.Timestamp(date),
    }


SYNTHETIC_ROWS = [
    # 1. Partial cancellation: buy 10, cancel 4 -> retain 6.
    make_row(1, "A", "I1", 10, 2.00, "2020-01-01"),
    make_row(1, "A", "C1", -4, 2.00, "2020-01-05"),

    # 2. Full cancellation followed by a later, unrelated purchase -> retain
    # only the later one. This is the exact defect a naive "sum the
    # quantities" approach gets wrong.
    make_row(2, "A", "I1", 10, 2.00, "2020-01-01"),
    make_row(2, "A", "C1", -10, 2.00, "2020-01-05"),
    make_row(2, "A", "I2", 5, 2.00, "2020-02-01"),

    # 3. Cancellation before any visible sale -> orphan, does not reach
    # forward and consume the future sale.
    make_row(3, "A", "C1", -8, 2.00, "2020-01-01"),
    make_row(3, "A", "I1", 5, 2.00, "2020-02-01"),

    # 4. Cancellation spanning multiple earlier invoices (FIFO across two lots).
    make_row(4, "A", "I1", 5, 2.00, "2020-01-01"),
    make_row(4, "A", "I2", 5, 2.00, "2020-01-10"),
    make_row(4, "A", "C1", -8, 2.00, "2020-01-15"),

    # 5. Same product at different prices -> FIFO cancels the older
    # (cheaper) lot first; price-compatible only touches the matching lot.
    make_row(5, "A", "I1", 5, 2.00, "2020-01-01"),
    make_row(5, "A", "I2", 5, 3.00, "2020-01-10"),
    make_row(5, "A", "C1", -6, 3.00, "2020-01-15"),

    # 6. One invoice with both a retained and a fully-cancelled product
    # (different StockCodes) -- netting is per (customer, item), not per invoice.
    make_row(6, "X", "I1", 5, 2.00, "2020-01-01"),
    make_row(6, "Y", "I1", 3, 4.00, "2020-01-01"),
    make_row(6, "X", "C1", -5, 2.00, "2020-01-05"),

    # 7. Same-timestamp tie: two sale rows for the same item at the exact
    # same moment -- a following cancellation should consume them in stable
    # (as-listed) order, deterministically.
    make_row(7, "A", "I1", 3, 2.00, "2020-01-01 10:00"),
    make_row(7, "A", "I2", 4, 2.00, "2020-01-01 10:00"),
    make_row(7, "A", "C1", -5, 2.00, "2020-01-05"),
]

EXPECTED_FIFO = {
    1: [6],
    2: [0, 5],
    3: [5],
    4: [0, 2],
    5: [0, 4],
    6: [0, 3],
    7: [0, 2],
}

# Cases 4, 5, and 7 deliberately diverge from FIFO -- price-compatible
# matching is LIFO-within-price-match, not strictly oldest-first.
EXPECTED_PRICE_COMPATIBLE = {
    1: [6],
    2: [0, 5],
    3: [5],
    4: [2, 0],
    5: [5, 0],
    6: [0, 3],
    7: [2, 0],
}


def _remaining_quantities(net_func):
    df = pd.DataFrame(SYNTHETIC_ROWS)
    remaining = (
        df.groupby(["Customer ID", "StockCode"], group_keys=False)
        .apply(net_func)
        .reindex(df.index)
    )
    df["remaining_quantity"] = remaining
    return df


@pytest.mark.parametrize("customer_id,expected", EXPECTED_FIFO.items())
def test_fifo_netting_matches_known_case(customer_id, expected):
    df = _remaining_quantities(net_group_fifo)
    actual = df.loc[
        (df["Customer ID"] == customer_id) & (df["Quantity"] > 0), "remaining_quantity"
    ].tolist()
    assert actual == expected


@pytest.mark.parametrize("customer_id,expected", EXPECTED_PRICE_COMPATIBLE.items())
def test_price_compatible_netting_matches_known_case(customer_id, expected):
    df = _remaining_quantities(net_group_price_compatible)
    actual = df.loc[
        (df["Customer ID"] == customer_id) & (df["Quantity"] > 0), "remaining_quantity"
    ].tolist()
    assert actual == expected


@pytest.mark.parametrize("customer_id", [1, 2, 3, 6])
def test_fifo_and_price_compatible_agree_without_price_ambiguity(customer_id):
    """Cases without multiple same-item lots at different prices must give
    identical answers regardless of matching policy -- the two methods only
    have a reason to disagree when price ambiguity actually exists."""
    fifo_df = _remaining_quantities(net_group_fifo)
    price_df = _remaining_quantities(net_group_price_compatible)

    fifo_values = fifo_df.loc[fifo_df["Customer ID"] == customer_id, "remaining_quantity"].tolist()
    price_values = price_df.loc[price_df["Customer ID"] == customer_id, "remaining_quantity"].tolist()
    assert fifo_values == price_values


def test_monetary_is_never_negative_after_netting():
    """Structural guarantee: since remaining_quantity is always >= 0 and
    Price is never negative in this dataset, NetRevenue (and therefore
    Monetary) can never go negative -- a customer's worst case is exactly 0."""
    df = _remaining_quantities(net_group_fifo)
    net_revenue = df["remaining_quantity"] * df["Price"]
    assert (net_revenue >= 0).all()
