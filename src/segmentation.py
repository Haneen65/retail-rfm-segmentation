"""Clustering utilities: preprocessing, K selection, profiling, naming."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]
SEED = 42


def preprocess(c: pd.DataFrame, features=FEATURES):
    """log1p to tame right skew from wholesalers, then StandardScaler (mean 0, var 1).

    CustomerID is never used as a feature (it is an identifier, not a quantity).
    """
    X = np.log1p(c[features].astype(float))
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def evaluate_k(Xs, ks=range(2, 11), sample=4000):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xs), size=min(sample, len(Xs)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
        rows.append(dict(k=k, WCSS=km.inertia_,
                         Silhouette=silhouette_score(Xs[idx], km.labels_[idx]),
                         DaviesBouldin=davies_bouldin_score(Xs, km.labels_)))
    out = pd.DataFrame(rows)
    # elbow: largest second difference of WCSS (the "kink")
    out["WCSS_drop"] = -out["WCSS"].diff()
    out["Kink"] = out["WCSS_drop"].diff().abs()
    return out


def stability(Xs, k, n_boot=15, frac=0.8):
    """Adjusted Rand Index between full-data KMeans and KMeans fit on random 80% subsets."""
    from sklearn.metrics import adjusted_rand_score
    base = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(Xs), size=int(frac * len(Xs)), replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e6))).fit(Xs[idx])
        scores.append(adjusted_rand_score(base.labels_[idx], km.predict(Xs[idx])))
    return float(np.mean(scores)), float(np.std(scores))


def profile(c: pd.DataFrame, label: str) -> pd.DataFrame:
    """Raw-unit profile per cluster, plus share of customers and revenue."""
    g = c.groupby(label)
    p = g.agg(
        Customers=("CustomerID", "count"),
        Recency_median=("Recency", "median"),
        Frequency_median=("Frequency", "median"),
        Monetary_median=("Monetary", "median"),
        Monetary_sum=("Monetary", "sum"),
        AOV_median=("AOV", "median"),
        BasketUnits_median=("AvgBasketUnits", "median"),
        ReturnRate_mean=("ReturnRate", "mean"),
        UK_share=("Is_UK", "mean"),
    )
    p["Customer_share_%"] = 100 * p["Customers"] / p["Customers"].sum()
    p["Revenue_share_%"] = 100 * p["Monetary_sum"] / p["Monetary_sum"].sum()
    return p.round(2)


def name_clusters(p: pd.DataFrame) -> dict:
    """Name clusters FROM their statistics, never from a template.

    Rules (all relative to the clusters themselves):
      * biggest revenue share                         -> "Power Core"
      * median frequency <= 1.5 and oldest recency    -> "Lapsed One-Timers"
      * of the rest: more recent than the median      -> "Fresh Explorers"
      * of the rest: less recent                      -> "Slipping Regulars"
    """
    names = {}
    power = p["Revenue_share_%"].idxmax()
    names[power] = "Power Core"
    rest = p.drop(index=power)
    one = rest[rest["Frequency_median"] <= 1.5]
    if len(one):
        names[one["Recency_median"].idxmax()] = "Lapsed One-Timers"
    rest = p.drop(index=list(names))
    if len(rest):
        cut = rest["Recency_median"].median()
        for k, row in rest.iterrows():
            names[k] = "Fresh Explorers" if row["Recency_median"] <= cut else "Slipping Regulars"
    return names


def dbscan_outliers(Xs, eps=0.5, min_samples=15):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
    return db.labels_
