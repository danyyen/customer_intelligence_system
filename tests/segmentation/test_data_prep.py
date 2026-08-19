"""Tests for StockCode categorization -- the rule set that decides what
counts as real merchandise versus operational noise (postage, fees, test
rows) before RFM/clustering ever sees the data.
"""

import pandas as pd

from customer_intelligence.data_prep import (
    apply_stockcode_categories,
    categorize_stock_code,
)


def test_operational_fee_codes():
    assert categorize_stock_code("POST") == "operational_fee"
    assert categorize_stock_code("BANK CHARGES") == "operational_fee"
    assert categorize_stock_code("AMAZONFEE") == "operational_fee"
    assert categorize_stock_code("DOT") == "operational_fee"


def test_test_dummy_codes():
    assert categorize_stock_code("TEST001") == "test_dummy"
    assert categorize_stock_code("TEST002") == "test_dummy"


def test_gift_voucher_codes():
    assert categorize_stock_code("gift_0001_20") == "gift_voucher"
    assert categorize_stock_code("gift_0001_50") == "gift_voucher"


def test_unclear_codes():
    assert categorize_stock_code("C3") == "unclear"
    assert categorize_stock_code("GIFT") == "unclear"


def test_standard_product_codes():
    """5 digits, optionally followed by 1-2 letters."""
    assert categorize_stock_code("85123A") == "product_standard"
    assert categorize_stock_code("21733") == "product_standard"
    assert categorize_stock_code("47503J") == "product_standard"


def test_alt_sku_product_codes():
    """Real merchandise under a different SKU convention -- must NOT be
    caught by the operational/test/unclear buckets, or real sales get
    silently dropped from the analysis."""
    assert categorize_stock_code("DCGS0058") == "product_alt_sku"
    assert categorize_stock_code("SP1002") == "product_alt_sku"


def test_apply_stockcode_categories_strips_whitespace():
    """A real bug found during development: '47503J ' (trailing space) and
    '47503J' were being treated as two different products."""
    df = pd.DataFrame({"StockCode": ["47503J ", "47503J", "POST"]})
    result = apply_stockcode_categories(df)

    assert result["StockCode"].tolist() == ["47503J", "47503J", "POST"]
    assert result.loc[0, "stockcode_category"] == "product_standard"
    assert result.loc[2, "stockcode_category"] == "operational_fee"
