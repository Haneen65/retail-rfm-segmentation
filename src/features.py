"""Customer-level features and RFM scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

RFM_SEGMENT_RULES = [
    # evaluated top-down, first match wins. Thresholds derived from the 1-5 quintile scores.
    ("Champions", lambda r: (r.R >= 4) & (r.F >= 4) & (r.M >= 4)),
    ("At-Risk High-Value", lambda r: (r.R <= 2) & (r.F >= 3) & (r.M >= 3)),
    ("Loyal Customers", lambda r: (r.R >= 3) & (r.F >= 3)),
    ("New Inflow", lambda r: (r.R >= 4) & (r.F <= 2)),
    ("Hibernating / Lost", lambda r: (r.R <= 2) & (r.F <= 2)),
]


def snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Day after the last transaction -> Recency is never 0 and is reproducible."""
    return df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def customer_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per identified customer.

    R  = days since last *purchase* invoice (returns do not reset recency)
    F  = unique purchase invoices
    M  = NET revenue (sales minus returns) - the guideline's "sum of all transaction revenues"
    """
    d = df[df["CustomerID"].notna() & ~df["IsAdjustment"] & ~df["IsZeroPrice"] & ~df["IsNonProduct"]].copy()
    snap = snapshot_date(df)
    buy = d[d["Quantity"] > 0]
    ret = d[d["Quantity"] < 0]

    g = buy.groupby("CustomerID")
    c = pd.DataFrame({
        "Recency": (snap - g["InvoiceDate"].max().dt.normalize()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "GrossRevenue": g["TotalLineRevenue"].sum(),
        "Units": g["Quantity"].sum(),
        "UniqueProducts": g["StockCode"].nunique(),
        "FirstPurchase": g["InvoiceDate"].min(),
        "Country": g["Country"].agg(lambda s: s.mode().iat[0]),
    })
    c["ReturnValue"] = -ret.groupby("CustomerID")["TotalLineRevenue"].sum()
    c["ReturnValue"] = c["ReturnValue"].fillna(0.0)
    c["Monetary"] = c["GrossRevenue"] - c["ReturnValue"]
    c["ReturnRate"] = (c["ReturnValue"] / c["GrossRevenue"]).clip(0, 1)
    c["AOV"] = c["GrossRevenue"] / c["Frequency"]
    c["AvgBasketUnits"] = c["Units"] / c["Frequency"]
    c["TenureDays"] = (snap - c["FirstPurchase"]).dt.days
    c["Is_UK"] = (c["Country"] == "United Kingdom").astype(int)
    c.index = c.index.astype(int)
    c.index.name = "CustomerID"
    return c.reset_index()


def score_rfm(c: pd.DataFrame) -> pd.DataFrame:
    """Quintile scores 1-5. R is inverted (5 = bought most recently).

    Frequency has huge ties (many one-time buyers) so we rank(method='first') before qcut;
    otherwise pd.qcut raises 'Bin edges must be unique'.
    """
    c = c.copy()
    c = c[c["Monetary"] > 1].copy()  # net-zero/negative customers (fully returned) cannot be scored/log-transformed
    c["R"] = pd.qcut(c["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    c["F"] = pd.qcut(c["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    c["M"] = pd.qcut(c["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    c["RFM_Score"] = c["R"].astype(str) + c["F"].astype(str) + c["M"].astype(str)
    c["RFM_Sum"] = c["R"] + c["F"] + c["M"]
    seg = pd.Series("Needs Attention", index=c.index)
    done = pd.Series(False, index=c.index)
    for name, rule in RFM_SEGMENT_RULES:
        m = rule(c) & ~done
        seg[m] = name
        done |= m
    c["RFM_Segment"] = seg
    return c


def dead_stock(df: pd.DataFrame, days: int = 90, top_q: float = 0.75) -> pd.DataFrame:
    """Products with high historical volume but zero/negative net sales in the final `days` days.

    'High historical volume' = units sold before the cutoff in the top quartile of all products.
    """
    p = df[df["IsProductSale"] | (df["IsCancellation"] & ~df["IsNonProduct"] & (df["UnitPrice"] > 0))]
    cutoff = p["InvoiceDate"].max() - pd.Timedelta(days=days)
    before, after = p[p["InvoiceDate"] <= cutoff], p[p["InvoiceDate"] > cutoff]
    hist = before.groupby("StockCode").agg(
        Description=("Description", lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""),
        HistUnits=("Quantity", "sum"), HistRevenue=("TotalLineRevenue", "sum"),
        LastSold=("InvoiceDate", "max"))
    recent = after.groupby("StockCode")["Quantity"].sum().rename("RecentNetUnits")
    out = hist.join(recent).fillna({"RecentNetUnits": 0})
    thr = out["HistUnits"].quantile(top_q)
    return out[(out["HistUnits"] >= thr) & (out["RecentNetUnits"] <= 0)].sort_values("HistRevenue", ascending=False)
