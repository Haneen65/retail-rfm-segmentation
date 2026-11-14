"""Streamlit dashboard — Online Retail customer segments.  Run: streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import features as ft, segmentation as sg, viz

ROOT = Path(__file__).parent
st.set_page_config(page_title="Retail Customer Galaxy", page_icon="🪐", layout="wide")

st.markdown("""<style>
.block-container{padding-top:1.6rem}
.kpi{background:#161b26;border:1px solid #232a3a;border-radius:14px;padding:14px 18px}
.kpi b{font-size:1.7rem;color:#f5b700;display:block}.kpi span{color:#9aa3b2;font-size:.82rem}
.play{border-left:4px solid var(--c);background:#161b26;border-radius:10px;padding:14px 18px;margin-bottom:10px}
.play h4{margin:0 0 4px 0}.play p{margin:2px 0;color:#c4cad6}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data…")
def load():
    p = ROOT / "data" / "processed"
    if not (p / "customers.parquet").exists():
        from src import pipeline
        pipeline.run()
    return (pd.read_parquet(p / "customers.parquet"), pd.read_parquet(p / "lines_clean.parquet"),
            pd.read_parquet(p / "orders.parquet"))


cust, lines, orders = load()

PLAYBOOK = {
    "Champions": ("Reward & recruit", "Loyalty perks, early access to the Q4 range, referral programme."),
    "Loyal Customers": ("Grow basket", "Cross-sell related catalogue segments; volume-tier pricing."),
    "At-Risk High-Value": ("Win back now", "Direct win-back e-mail + time-limited offer; account manager call for the biggest."),
    "New Inflow": ("Second order", "Welcome sequence with a discount that unlocks on order #2 within 3 weeks."),
    "Needs Attention": ("Monitor", "Light-touch nurture; re-score monthly."),
    "Hibernating / Lost": ("Cheap reactivation", "One automated e-mail, then suppress. No paid media."),
    "Power Core": ("Protect", "Dedicated account management, August pre-season preview, no blanket discounts."),
    "Slipping Regulars": ("Highest ROI", "Trigger win-back when a regular passes ~45 days; bundle / volume offers."),
    "Fresh Explorers": ("Convert", "Cross-sell from the first basket; nudge a 2nd and 3rd order."),
    "Lapsed One-Timers": ("Low-cost tail", "A single reactivation e-mail, then let go."),
}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🪐 Controls")
view = st.sidebar.radio("Segmentation lens", ["K-Means clusters", "RFM rules"], help="Two ways of cutting the same customers.")
col, colors = (("ClusterName", viz.CLUSTER_COLORS) if view == "K-Means clusters" else ("RFM_Segment", viz.SEG_COLORS))
chosen = st.sidebar.multiselect("Segments", list(colors), default=list(colors))
countries = ["All"] + sorted(cust["Country"].unique())
country = st.sidebar.selectbox("Country", countries)
rmax = int(cust["Recency"].max())
rrange = st.sidebar.slider("Recency (days since last order)", 1, rmax, (1, rmax))
hide_whales = st.sidebar.toggle("Hide DBSCAN outliers (whales)", False)

f = cust[cust[col].isin(chosen) & cust["Recency"].between(*rrange)]
if country != "All":
    f = f[f["Country"] == country]
if hide_whales:
    f = f[~f["DBSCAN_Outlier"]]

st.title("Retail Customer Galaxy")
st.caption("Online Retail · 1 Dec 2010 – 9 Dec 2011 · RFM + K-Means (k=4) · all figures net of returns")

if f.empty:
    st.warning("No customers match these filters.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
kpi = lambda c, v, l: c.markdown(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)
kpi(k1, f"{len(f):,}", "customers in view")
kpi(k2, f"£{f.Monetary.sum()/1e6:,.2f}M", f"net revenue ({100*f.Monetary.sum()/cust.Monetary.sum():.0f}% of total)")
kpi(k3, f"{f.Frequency.median():.0f}", "median orders / customer")
kpi(k4, f"{f.Recency.median():.0f} d", "median days since last order")
kpi(k5, f"£{f.Monetary.median():,.0f}", "median customer value")
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["🪐 Segments", "🎯 Playbook", "📈 Business pulse", "🔎 Customer lookup", "📐 Method"])

with t1:
    a, b = st.columns([3, 2])
    a.plotly_chart(viz.galaxy(f, col, colors), width="stretch")
    b.plotly_chart(viz.value_vs_size(f, col, colors), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.radar(f, col, colors), width="stretch")
    c2.plotly_chart(viz.segment_treemap(f, col, colors), width="stretch")
    if view == "K-Means clusters":
        st.plotly_chart(viz.sankey(cust), width="stretch")
    st.subheader("Segment profile")
    prof = sg.profile(f, col).sort_values("Revenue_share_%", ascending=False)
    st.dataframe(prof, width="stretch")

with t2:
    st.subheader("What to do with each segment")
    share = f.groupby(col)["Monetary"].agg(["count", "sum"])
    for seg in chosen:
        if seg not in share.index:
            continue
        head, text = PLAYBOOK[seg]
        n, rev = share.loc[seg]
        st.markdown(f'<div class="play" style="--c:{colors[seg]}"><h4>{seg} — {head}</h4>'
                    f'<p>{text}</p><p><b>{int(n):,}</b> customers · <b>£{rev:,.0f}</b> net revenue</p></div>', unsafe_allow_html=True)
    st.download_button("⬇️ Download these customers (CSV)",
                       f[["CustomerID", "Country", "Recency", "Frequency", "Monetary", "RFM_Score", "RFM_Segment", "ClusterName"]]
                       .to_csv(index=False).encode(), "segment_customers.csv", "text/csv")

with t3:
    st.plotly_chart(viz.monthly_revenue(lines), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(viz.hour_weekday_heatmap(orders), width="stretch")
    b.plotly_chart(viz.pareto(lines), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(viz.top_products(lines, "Revenue"), width="stretch")
    b.plotly_chart(viz.country_chart(lines), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(viz.returns_by_product(lines), width="stretch")
    b.plotly_chart(viz.cohort_heatmap(lines), width="stretch")
    with st.expander("Dead-stock watch-list (top-quartile volume, nothing sold in last 90 days)"):
        st.dataframe(ft.dead_stock(lines).head(25), width="stretch")

with t4:
    cid = st.number_input("CustomerID", min_value=int(cust.CustomerID.min()), max_value=int(cust.CustomerID.max()),
                          value=int(cust.sort_values("Monetary").CustomerID.iat[-1]))
    row = cust[cust.CustomerID == cid]
    if row.empty:
        st.info("That ID is not among the scored customers (guest, or net-zero after returns).")
    else:
        r = row.iloc[0]
        st.markdown(f"### Customer {cid} · {r.Country}")
        a, b, c, d = st.columns(4)
        a.metric("RFM score", r.RFM_Score, r.RFM_Segment)
        b.metric("K-Means cluster", r.ClusterName)
        c.metric("Orders / last order", f"{r.Frequency} / {r.Recency} d ago")
        d.metric("Net revenue", f"£{r.Monetary:,.0f}")
        head, text = PLAYBOOK[r.ClusterName]
        st.info(f"**Suggested action — {head}:** {text}")
        if r.Whale or r.DBSCAN_Outlier:
            st.warning("Flagged as a statistical outlier / whale → manage as a named account.")
        hist = lines[(lines.CustomerID == cid) & lines.IsProductSale].groupby("YearMonth")["TotalLineRevenue"].sum().reset_index()
        st.plotly_chart(px.bar(hist, x="YearMonth", y="TotalLineRevenue", color_discrete_sequence=["#f5b700"]), width="stretch")

with t5:
    st.markdown("""
**Pipeline** — dedupe → flag (never silently delete) returns, zero-price, fees, guests → net revenue → customer table →
RFM quintiles (R inverted: **5 = most recent**) → `log1p` + `StandardScaler` → K-Means.

**Why K=4** — elbow kink at 4, silhouette ≈ 0.34 (same as k=3), bootstrap ARI ≈ 0.94, and four distinct marketing playbooks.
Silhouette 0.34 is *moderate*: behaviour is a continuum, so segment borders are soft.

**Caveats** — one year of data; descriptive not causal; ~15% of product revenue is guest checkout and cannot be segmented;
whales (top 1% / DBSCAN noise) are flagged because they distort centroids.
""")
