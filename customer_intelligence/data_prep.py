"""Load and clean the UCI Online Retail II transaction data down to the
customer-attributed merchandise population used for RFM feature engineering.

Pipeline: load_transactions -> tag_invoice_type -> build_model_df.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_SHEETS = ("Year 2009-2010", "Year 2010-2011")

# --- StockCode categorization -------------------------------------------
#
# Real product codes in this dataset are, overwhelmingly, 5 digits optionally
# followed by 1-2 letters (e.g. "85123A"). Everything else was audited by
# hand against the actual data before being bucketed below -- see the
# project notebook's "StockCode Audit" section for how each set was derived.
STANDARD_CODE_PATTERN = r"^\d{5}[A-Za-z]{0,2}$"

OPERATIONAL_CODES = {
    "POST", "DOT", "M", "m", "C2", "D", "S", "BANK CHARGES",
    "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK",
}
TEST_CODES = {"TEST001", "TEST002"}
UNCLEAR_CODES = {"C3", "GIFT"}  # single-row, no description, no clear meaning

# Dropped before RFM/clustering: none of these represent a customer purchase.
EXCLUDE_STOCKCODE_CATEGORIES = {"operational_fee", "test_dummy", "unclear"}


def load_transactions(path: Path, sheets: Iterable[str] = DEFAULT_SHEETS) -> pd.DataFrame:
    """Read each sheet of the UCI Online Retail II workbook separately, tag it
    with its source sheet, and concatenate into one DataFrame."""
    frames = []
    for sheet_name in sheets:
        sheet_df = pd.read_excel(path, sheet_name=sheet_name)
        sheet_df["source_sheet"] = sheet_name
        frames.append(sheet_df)
    return pd.concat(frames, ignore_index=True)


def tag_invoice_type(df: pd.DataFrame) -> pd.DataFrame:
    """Classify every row by its Invoice prefix: 'sale' (no prefix, normal
    invoice number), 'cancellation' (C-prefix), or 'adjustment' (A-prefix,
    bad-debt write-offs -- not real transactions)."""
    df = df.copy()
    invoice_prefix = df["Invoice"].astype(str).str.extract(r"^([A-Za-z]+)")[0]
    df["invoice_type"] = np.select(
        [invoice_prefix.isna(), invoice_prefix == "C", invoice_prefix == "A"],
        ["sale", "cancellation", "adjustment"],
        default="other",
    )
    return df


def categorize_stock_code(code: str) -> str:
    """Bucket a single StockCode into one of: operational_fee, test_dummy,
    unclear, gift_voucher, product_standard, product_alt_sku.

    product_alt_sku exists because some real products (e.g. the DCGS*/SP*
    ranges) use a letter-prefixed SKU convention instead of the usual
    5-digit format -- they are genuine merchandise, not junk, despite not
    matching STANDARD_CODE_PATTERN.
    """
    if code in OPERATIONAL_CODES:
        return "operational_fee"
    if code in TEST_CODES:
        return "test_dummy"
    if code in UNCLEAR_CODES:
        return "unclear"
    if code.startswith("gift_0001_"):
        return "gift_voucher"
    if re.match(STANDARD_CODE_PATTERN, code):
        return "product_standard"
    return "product_alt_sku"


def apply_stockcode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray whitespace from StockCode (fixes real duplicates like
    '47503J ' vs '47503J') and attach a stockcode_category column."""
    df = df.copy()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    code_to_category = {code: categorize_stock_code(code) for code in df["StockCode"].unique()}
    df["stockcode_category"] = df["StockCode"].map(code_to_category)
    return df


def build_model_df(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline from a `tag_invoice_type`-tagged raw DataFrame
    down to the customer-attributed merchandise population used for RFM and
    clustering:

    1. Drop 'adjustment' invoices (bad-debt write-offs, not transactions).
    2. Drop non-merchandise StockCode categories (operational fees, test
       rows, unclear single-row codes) -- real product sales and
       cancellations are both kept, cancellations are netted out later.
    3. Drop rows with no Customer ID (unattributable to any customer).
    """
    gross_df = df[df["invoice_type"] != "adjustment"].reset_index(drop=True)
    gross_df = apply_stockcode_categories(gross_df)

    model_df = gross_df[~gross_df["stockcode_category"].isin(EXCLUDE_STOCKCODE_CATEGORIES)].reset_index(drop=True)

    model_df = model_df[model_df["Customer ID"].notna()].reset_index(drop=True)
    model_df["Customer ID"] = model_df["Customer ID"].astype(int)

    return model_df
