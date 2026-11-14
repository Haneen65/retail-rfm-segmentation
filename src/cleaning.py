"""Data engineering for the Online Retail dataset.

Every function documents the *business reason* behind the rule, because in this
project a cleaning step is only acceptable if we can explain its revenue impact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Stock codes that are services / fees / manual adjustments, NOT physical products.
# Keeping them would pollute "top products" and basket-size statistics.
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "m", "C2", "D", "S", "B", "BANK CHARGES", "AMAZONFEE",
    "CRUK", "ADJUST", "ADJUST2", "PADS", "TEST001", "TEST002",
}


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw file (parquet cache or the original xlsx)."""
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """One-table data-quality audit used in notebook 01."""
    inv = df["InvoiceNo"]
    rows = {
        "Rows": len(df),
        "Exact duplicate rows": int(df.duplicated().sum()),
        "Missing CustomerID": int(df["CustomerID"].isna().sum()),
        "Missing Description": int(df["Description"].isna().sum()),
        "Cancellation lines (InvoiceNo starts with C)": int(inv.str.startswith("C").sum()),
        "Adjustment invoices (InvoiceNo starts with A)": int(inv.str.startswith("A").sum()),
        "Negative quantity lines": int((df["Quantity"] < 0).sum()),
        "Negative quantity but NOT a cancellation": int(((df["Quantity"] < 0) & ~inv.str.startswith("C")).sum()),
        "UnitPrice == 0": int((df["UnitPrice"] == 0).sum()),
        "UnitPrice < 0": int((df["UnitPrice"] < 0).sum()),
        "Non-product stock codes (fees/postage/...)": int(df["StockCode"].isin(NON_PRODUCT_CODES).sum()),
    }
    return pd.Series(rows, name="count").to_frame()


def guest_checkout_check(df: pd.DataFrame) -> dict:
    """Guideline: group missing-CustomerID rows by InvoiceNo before isolating them.

    If every invoice is either fully identified or fully anonymous, the missing IDs
    are guest-checkout *blocks* (a real behaviour) and not random data loss.
    """
    g = df.groupby("InvoiceNo")["CustomerID"].agg(lambda s: s.isna().mean())
    return {
        "invoices_total": int(len(g)),
        "invoices_fully_guest": int((g == 1).sum()),
        "invoices_fully_identified": int((g == 0).sum()),
        "invoices_mixed": int(((g > 0) & (g < 1)).sum()),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (lines, log).

    `lines` keeps *every* transaction line that survives de-duplication, with flags,
    so downstream code chooses what to include (sales-only vs. net of returns).
    `log` records each decision and its revenue impact.
    """
    log = []
    df = df.copy()
    base_rev = (df["Quantity"] * df["UnitPrice"]).sum()

    # ---- types & time features -------------------------------------------------
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Date"] = df["InvoiceDate"].dt.normalize()

    # ---- exact duplicates ------------------------------------------------------
    n = len(df)
    dup = df.duplicated(keep="first")
    dup_rev = (df.loc[dup, "Quantity"] * df.loc[dup, "UnitPrice"]).sum()
    df = df.loc[~dup].copy()
    log.append(("Remove exact duplicate lines", n - len(df), dup_rev,
                "Same invoice, product, qty, price, time -> scanner/export duplication, would double-count revenue."))

    # ---- flags -----------------------------------------------------------------
    inv = df["InvoiceNo"]
    df["IsCancellation"] = inv.str.startswith("C")
    df["IsAdjustment"] = inv.str.startswith("A")
    df["IsNonProduct"] = df["StockCode"].isin(NON_PRODUCT_CODES)
    df["IsZeroPrice"] = df["UnitPrice"] == 0
    df["IsNegPrice"] = df["UnitPrice"] < 0
    df["IsGuest"] = df["CustomerID"].isna()
    df["IsNegQtyNotCancel"] = (df["Quantity"] < 0) & ~df["IsCancellation"]
    df["Is_UK"] = (df["Country"] == "United Kingdom").astype(int)

    # ---- feature: line revenue --------------------------------------------------
    df["TotalLineRevenue"] = df["Quantity"] * df["UnitPrice"]

    for name, mask, why in [
        ("Flag: bad-debt adjustments (A*, negative price)", df["IsAdjustment"] | df["IsNegPrice"],
         "Accounting entries, not customer purchases -> excluded from behaviour analysis."),
        ("Flag: zero-price lines", df["IsZeroPrice"],
         "Giveaways / write-offs / system errors -> no revenue, must not count as a purchase."),
        ("Flag: negative qty without 'C' prefix", df["IsNegQtyNotCancel"],
         "Damages, lost stock, manual corrections (mostly guests, price 0)."),
        ("Flag: non-product codes (postage, fees, manual)", df["IsNonProduct"],
         "Real money, but not product demand -> kept in revenue, removed from product rankings."),
        ("Flag: cancellations (C*)", df["IsCancellation"],
         "Returns. KEPT and netted against revenue instead of being dropped."),
        ("Flag: guest checkout (no CustomerID)", df["IsGuest"],
         "Entire invoices are anonymous -> real sales, but cannot enter customer-level RFM."),
    ]:
        log.append((name, int(mask.sum()), df.loc[mask, "TotalLineRevenue"].sum(), why))

    # a clean "analysis" flag: genuine customer-product lines
    df["IsProductSale"] = (
        ~df["IsCancellation"] & ~df["IsAdjustment"] & ~df["IsNonProduct"]
        & ~df["IsZeroPrice"] & ~df["IsNegPrice"] & (df["Quantity"] > 0)
    )
    log.append(("Genuine product sale lines (Qty>0, price>0, product code)", int(df["IsProductSale"].sum()),
                df.loc[df["IsProductSale"], "TotalLineRevenue"].sum(), "Base for product & basket analytics."))

    log_df = pd.DataFrame(log, columns=["Decision", "Rows affected", "Revenue impact (GBP)", "Business reason"])
    log_df.attrs["raw_net_revenue"] = base_rev
    return df.reset_index(drop=True), log_df


def match_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to match each cancellation line to an earlier purchase.

    Rule: same CustomerID + StockCode, with an earlier sale whose quantity is >= returned qty.
    Returns the cancellation lines with a `Matched` column.
    """
    sales = df[(df["Quantity"] > 0) & df["CustomerID"].notna() & ~df["IsAdjustment"]]
    canc = df[df["IsCancellation"] & df["CustomerID"].notna()].copy()
    canc["ReturnQty"] = -canc["Quantity"]
    first = sales.groupby(["CustomerID", "StockCode"]).agg(
        first_purchase=("InvoiceDate", "min"), bought_qty=("Quantity", "sum")).reset_index()
    m = canc.merge(first, on=["CustomerID", "StockCode"], how="left")
    m["Matched"] = (m["first_purchase"].notna() & (m["first_purchase"] <= m["InvoiceDate"])
                    & (m["bought_qty"] >= m["ReturnQty"]))
    return m


def order_table(df: pd.DataFrame) -> pd.DataFrame:
    """Invoice-level table (guideline: do not treat every row as an order)."""
    s = df[df["IsProductSale"]]
    o = s.groupby("InvoiceNo").agg(
        InvoiceDate=("InvoiceDate", "first"), CustomerID=("CustomerID", "first"),
        Country=("Country", "first"), Is_UK=("Is_UK", "first"),
        Revenue=("TotalLineRevenue", "sum"), Units=("Quantity", "sum"),
        Lines=("StockCode", "nunique"), Hour=("Hour", "first"),
        DayOfWeek=("DayOfWeek", "first"), YearMonth=("YearMonth", "first"),
    ).reset_index()
    o["IsGuest"] = o["CustomerID"].isna()
    return o
