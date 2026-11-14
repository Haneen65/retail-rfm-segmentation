"""End-to-end reproducible pipeline: raw parquet/xlsx -> processed tables used by notebooks, app and report.

Run:  python -m src.pipeline [path-to-raw]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import cleaning as cl, features as ft, segmentation as sg

ROOT = Path(__file__).resolve().parents[1]
K_FINAL = 4  # justified in notebook 02 (silhouette plateau, elbow kink, ARI stability, actionability)


def run(raw_path: str | None = None):
    raw_path = raw_path or str(ROOT / "data" / "online_retail_raw.parquet")
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    raw = cl.load_raw(raw_path)
    df, log = cl.clean(raw)
    log.to_csv(out / "cleaning_log.csv", index=False)
    df.to_parquet(out / "lines_clean.parquet")

    c = ft.score_rfm(ft.customer_table(df))
    Xs, _ = sg.preprocess(c)
    km = KMeans(n_clusters=K_FINAL, n_init=20, random_state=sg.SEED).fit(Xs)
    c["Cluster"] = km.labels_
    names = sg.name_clusters(sg.profile(c, "Cluster"))
    c["ClusterName"] = c["Cluster"].map(names)

    noise = sg.dbscan_outliers(Xs, eps=0.4, min_samples=10)
    c["DBSCAN_Outlier"] = noise == -1
    c["Whale"] = c["Monetary"] >= c["Monetary"].quantile(0.99)

    # 3-D coordinates (scaled log-RFM) for the "customer galaxy"
    c[["zR", "zF", "zM"]] = Xs
    c.to_parquet(out / "customers.parquet")

    orders = cl.order_table(df)
    orders.to_parquet(out / "orders.parquet")
    meta = {"snapshot": str(ft.snapshot_date(df).date()), "k": K_FINAL, "n_customers": int(len(c))}
    (out / "meta.json").write_text(json.dumps(meta))
    return df, c, orders, log


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
    print("pipeline done")
