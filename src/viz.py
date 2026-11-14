"""Plotly figure factory shared by the notebooks, the Streamlit app and the HTML report.

Colours are theme-neutral (mid-tones, transparent backgrounds) so the same figure reads on
light and dark pages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEG_COLORS = {
    "Champions": "#f5b700", "Loyal Customers": "#2ec4b6", "At-Risk High-Value": "#ff6b6b",
    "New Inflow": "#5dade2", "Needs Attention": "#a78bfa", "Hibernating / Lost": "#8d99ae",
}
CLUSTER_COLORS = {
    "Power Core": "#f5b700", "Slipping Regulars": "#ff6b6b",
    "Fresh Explorers": "#2ec4b6", "Lapsed One-Timers": "#8d99ae",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GREY = "#8a8f98"


def _style(fig, title=None, height=430):
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        height=height, margin=dict(l=50, r=20, t=60 if title else 20, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=GREY),
        legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


# ------------------------------------------------------------------ time
def monthly_revenue(df):
    d = df[~df["IsAdjustment"] & ~df["IsZeroPrice"]]
    m = d.groupby("YearMonth").agg(Revenue=("TotalLineRevenue", "sum")).reset_index()
    inv = df[df["IsProductSale"]].groupby("YearMonth")["InvoiceNo"].nunique().rename("Orders").reset_index()
    m = m.merge(inv, on="YearMonth")
    colors = ["#f5b700" if ym in ("2011-09", "2011-10", "2011-11") else "#5b8def" for ym in m["YearMonth"]]
    colors[-1] = "rgba(91,141,239,.35)"  # December 2011 is partial (data ends 9 Dec)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=m["YearMonth"], y=m["Revenue"], marker_color=colors, name="Net revenue (£)",
                hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=m["YearMonth"], y=m["Orders"], name="Orders", mode="lines+markers",
                    line=dict(color="#ff6b6b", width=2), secondary_y=True,
                    hovertemplate="%{x}<br>%{y:,} orders<extra></extra>")
    fig.add_annotation(x="2011-11", y=m.loc[m.YearMonth == "2011-11", "Revenue"].iat[0], text="Q4 surge",
                       showarrow=True, arrowhead=2, ay=-40, font=dict(color="#f5b700"))
    fig.add_annotation(x="2011-12", y=m["Revenue"].iat[-1], text="partial month<br>(data ends 9 Dec)",
                       showarrow=True, ay=-50, font=dict(size=10))
    fig.update_yaxes(title_text="Net revenue (£)", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True, showgrid=False)
    return _style(fig, "Monthly net revenue — the Q4 holiday surge")


def hour_weekday_heatmap(orders):
    z = (orders.groupby(["DayOfWeek", "Hour"])["Revenue"].sum().unstack(fill_value=0)
         .reindex(DAYS).dropna(how="all"))
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale="Viridis",
                               hovertemplate="%{y} %{x}:00<br>£%{z:,.0f}<extra></extra>",
                               colorbar=dict(title="£")))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Hour of day")
    return _style(fig, "When the money arrives — revenue by weekday × hour", 380)


def weekday_bar(orders):
    w = orders.groupby("DayOfWeek").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "count")).reindex(DAYS).dropna()
    fig = px.bar(w.reset_index(), x="DayOfWeek", y="Revenue", color_discrete_sequence=["#5b8def"])
    return _style(fig, "Revenue by weekday (no Saturday trading)", 340)


# ------------------------------------------------------------------ products
def net_product_lines(df):
    """Product sales AND product returns, so rankings are NET of cancellations.

    Why: the #1 gross-revenue product (80,995 'Paper Craft, Little Birdie') was fully cancelled the
    same day; ranking on gross sales would crown a phantom bestseller.
    """
    return df[df["IsProductSale"] | (df["IsCancellation"] & ~df["IsNonProduct"] & (df["UnitPrice"] > 0))]


def top_products(df, by="Revenue", n=12):
    s = net_product_lines(df)
    p = s.groupby("StockCode").agg(Description=("Description", lambda x: x.mode().iat[0]),
                                    Units=("Quantity", "sum"), Revenue=("TotalLineRevenue", "sum"))
    p = p.sort_values(by, ascending=False).head(n).iloc[::-1]
    p["Label"] = p["Description"].str.title().str.slice(0, 34)
    fig = go.Figure(go.Bar(x=p[by], y=p["Label"], orientation="h",
                           marker_color="#2ec4b6" if by == "Revenue" else "#a78bfa",
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    return _style(fig, f"Top {n} products by net {by.lower()} (after returns)", 430)


def pareto(df):
    s = net_product_lines(df).groupby("StockCode")["TotalLineRevenue"].sum()
    s = s[s > 0].sort_values(ascending=False)
    cum = s.cumsum() / s.sum()
    x = np.arange(1, len(s) + 1) / len(s) * 100
    n80 = int((cum < 0.8).sum() + 1)
    pct80 = n80 / len(s) * 100
    fig = go.Figure()
    fig.add_scatter(x=x, y=cum.values * 100, mode="lines", fill="tozeroy", line=dict(color="#f5b700", width=3),
                    fillcolor="rgba(245,183,0,.15)", name="Cumulative revenue",
                    hovertemplate="Top %{x:.1f}% of products<br>= %{y:.1f}% of revenue<extra></extra>")
    fig.add_scatter(x=[20, 20], y=[0, 100], mode="lines", line=dict(color=GREY, dash="dot"), showlegend=False)
    fig.add_scatter(x=[pct80], y=[80], mode="markers+text", marker=dict(size=12, color="#ff6b6b"),
                    text=[f"80% of revenue from just {pct80:.0f}% of products"], textposition="bottom right", showlegend=False)
    fig.update_xaxes(title="% of products (ranked by revenue)", range=[0, 100])
    fig.update_yaxes(title="% of cumulative revenue", range=[0, 101])
    fig = _style(fig, "Pareto: how concentrated is revenue?", 400)
    fig.layout.meta = {"n80": n80, "pct80": pct80, "n_products": len(s)}
    return fig


# ------------------------------------------------------------------ geography
def country_chart(df, exclude_uk=True):
    s = df[df["IsProductSale"]]
    g = s.groupby("Country").agg(Revenue=("TotalLineRevenue", "sum"), Orders=("InvoiceNo", "nunique"),
                                 Customers=("CustomerID", "nunique")).reset_index()
    if exclude_uk:
        g = g[g["Country"] != "United Kingdom"]
    g = g.sort_values("Revenue", ascending=False).head(15)
    fig = px.treemap(g, path=["Country"], values="Revenue", color="Customers",
                     color_continuous_scale="Tealgrn", hover_data=["Orders"])
    fig.update_traces(texttemplate="<b>%{label}</b><br>£%{value:,.0f}")
    return _style(fig, "International revenue (UK excluded) — size = revenue, colour = customers", 430)


# ------------------------------------------------------------------ baskets
def basket_hist(orders):
    o = orders[~orders["IsGuest"] | True]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Units per invoice (log scale)", "Order value £ (log scale)"))
    fig.add_histogram(x=np.log10(o["Units"].clip(lower=1)), nbinsx=50, marker_color="#5b8def", name="Units", row=1, col=1)
    fig.add_histogram(x=np.log10(o["Revenue"].clip(lower=1)), nbinsx=50, marker_color="#2ec4b6", name="Value", row=1, col=2)
    for c, vals in ((1, [1, 10, 100, 1000, 10000]), (2, [1, 10, 100, 1000, 10000, 100000])):
        fig.update_xaxes(tickvals=np.log10(vals), ticktext=[f"{v:,}" for v in vals], row=1, col=c)
    fig.update_layout(showlegend=False)
    return _style(fig, "Basket size and order value are heavily right-skewed → wholesale signal", 360)


# ------------------------------------------------------------------ returns
def returns_by_product(df, n=12):
    r = df[df["IsCancellation"] & ~df["IsNonProduct"] & (df["UnitPrice"] > 0)]
    p = r.groupby("StockCode").agg(Description=("Description", lambda x: x.mode().iat[0] if x.notna().any() else ""),
                                   Units=("Quantity", lambda x: -x.sum()), Value=("TotalLineRevenue", lambda x: -x.sum()))
    p = p.sort_values("Value", ascending=False).head(n).iloc[::-1]
    p["Label"] = p["Description"].str.title().str.slice(0, 34)
    fig = go.Figure(go.Bar(x=p["Value"], y=p["Label"], orientation="h", marker_color="#ff6b6b",
                           hovertemplate="%{y}<br>£%{x:,.0f} returned<extra></extra>"))
    return _style(fig, "Most-returned products by value", 430)


def returns_over_time(df):
    d = df[~df["IsAdjustment"] & ~df["IsZeroPrice"] & ~df["IsNonProduct"]]
    s = d[d["Quantity"] > 0].groupby("YearMonth")["TotalLineRevenue"].sum()
    r = -d[d["Quantity"] < 0].groupby("YearMonth")["TotalLineRevenue"].sum()
    rate = (r / s * 100).dropna()
    fig = go.Figure(go.Scatter(x=rate.index, y=rate.values, mode="lines+markers", line=dict(color="#ff6b6b", width=3)))
    fig.update_yaxes(title="Returns as % of gross sales")
    return _style(fig, "Return rate by month", 340)


# ------------------------------------------------------------------ cohorts
def cohort_heatmap(df):
    d = df[df["IsProductSale"] & df["CustomerID"].notna()].copy()
    d["Cohort"] = d.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
    d["Period"] = d["InvoiceDate"].dt.to_period("M")
    d["Age"] = (d["Period"] - d["Cohort"]).apply(lambda x: x.n)
    t = d.groupby(["Cohort", "Age"])["CustomerID"].nunique().unstack(fill_value=0)
    ret = t.div(t[0], axis=0) * 100
    ret.index = ret.index.astype(str)
    z = ret.where(np.triu(np.ones(ret.shape)[:, ::-1]).astype(bool) | True)
    # mask impossible cells (future months) -> NaN
    n = len(ret)
    for i in range(n):
        z.iloc[i, n - i:] = np.nan
    fig = go.Figure(go.Heatmap(z=z.values, x=[f"M+{c}" for c in z.columns], y=z.index, colorscale="YlGnBu",
                               hovertemplate="Cohort %{y}<br>%{x}: %{z:.0f}% still buying<extra></extra>",
                               colorbar=dict(title="%")))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Cohort retention — the Dec-2010 cohort is unusually sticky", 480)


# ------------------------------------------------------------------ RFM
def rfm_heatmap(c):
    g = c.groupby(["R", "F"])["Monetary"].median().unstack()
    n = c.groupby(["R", "F"]).size().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(z=g.values, x=[f"F{x}" for x in g.columns], y=[f"R{x}" for x in g.index],
                               colorscale="Magma", customdata=n.values,
                               hovertemplate="%{y} · %{x}<br>median £%{z:,.0f}<br>%{customdata} customers<extra></extra>",
                               colorbar=dict(title="median £")))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "RFM grid — median spend by Recency × Frequency score (R5 = most recent)", 400)


def segment_treemap(c, col="RFM_Segment", colors=SEG_COLORS):
    g = c.groupby(col).agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum")).reset_index()
    fig = px.treemap(g, path=[col], values="Customers", color=col, color_discrete_map=colors, custom_data=["Revenue"])
    fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,} customers<br>£%{customdata[0]:,.0f}",
                      hovertemplate="%{label}<br>%{value} customers<br>£%{customdata[0]:,.0f} revenue<extra></extra>")
    return _style(fig, "Segments sized by customers — hover for revenue", 400)


def value_vs_size(c, col="RFM_Segment", colors=SEG_COLORS):
    g = c.groupby(col).agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum")).reset_index()
    g["Customers_%"] = 100 * g["Customers"] / g["Customers"].sum()
    g["Revenue_%"] = 100 * g["Revenue"] / g["Revenue"].sum()
    g = g.sort_values("Revenue_%", ascending=False)
    fig = go.Figure()
    fig.add_bar(x=g[col], y=g["Customers_%"], name="% of customers", marker_color="#8a8f98")
    fig.add_bar(x=g[col], y=g["Revenue_%"], name="% of revenue", marker_color="#f5b700")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="%")
    return _style(fig, "Who pays the bills? Share of customers vs share of revenue", 380)


def galaxy(c, color="ClusterName", colors=CLUSTER_COLORS):
    fig = px.scatter_3d(c, x="zR", y="zF", z="zM", color=color, color_discrete_map=colors, opacity=0.75,
                        hover_data={"CustomerID": True, "Recency": True, "Frequency": True, "Monetary": ":,.0f",
                                    "zR": False, "zF": False, "zM": False})
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(scene=dict(xaxis_title="Recency (z, log)", yaxis_title="Frequency (z, log)",
                                 zaxis_title="Monetary (z, log)",
                                 xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(128,128,128,.25)", zerolinecolor="rgba(128,128,128,.4)"),
                                 yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(128,128,128,.25)", zerolinecolor="rgba(128,128,128,.4)"),
                                 zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(128,128,128,.25)", zerolinecolor="rgba(128,128,128,.4)")),
                      legend_title_text="Segment")
    return _style(fig, "The customer galaxy — every dot is a customer (drag to rotate)", 560)


def sankey(c):
    left = list(SEG_COLORS)
    right = list(CLUSTER_COLORS)
    ct = pd.crosstab(c["RFM_Segment"], c["ClusterName"]).reindex(index=left, columns=right, fill_value=0)
    src, tgt, val = [], [], []
    for i, l in enumerate(left):
        for j, r in enumerate(right):
            if ct.loc[l, r] > 0:
                src.append(i); tgt.append(len(left) + j); val.append(int(ct.loc[l, r]))
    cols = [SEG_COLORS[l] for l in left] + [CLUSTER_COLORS[r] for r in right]
    fig = go.Figure(go.Sankey(node=dict(label=left + right, color=cols, pad=14, thickness=16),
                              link=dict(source=src, target=tgt, value=val,
                                        color=[cols[s].replace("#", "#") + "66" if False else "rgba(138,143,152,.35)" for s in src])))
    return _style(fig, "Rule-based RFM segments  →  data-driven K-Means clusters", 430)


def radar(c, col="ClusterName", colors=CLUSTER_COLORS):
    feats = {"Recency (recent=high)": -np.log1p(c["Recency"]), "Frequency": np.log1p(c["Frequency"]),
             "Monetary": np.log1p(c["Monetary"]), "Avg basket": np.log1p(c["AvgBasketUnits"]),
             "Product variety": np.log1p(c["UniqueProducts"])}
    z = pd.DataFrame(feats)
    z = (z - z.min()) / (z.max() - z.min())
    z[col] = c[col].values
    m = z.groupby(col).mean()
    fig = go.Figure()
    for name, row in m.iterrows():
        fig.add_scatterpolar(r=list(row.values) + [row.values[0]], theta=list(m.columns) + [m.columns[0]],
                             fill="toself", name=name, line=dict(color=colors.get(name)), opacity=0.55)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)))
    return _style(fig, "Segment fingerprints (normalised log-features)", 460)


def elbow_silhouette(ev):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=ev["k"], y=ev["WCSS"], name="WCSS (elbow)", mode="lines+markers", line=dict(color="#5b8def", width=3))
    fig.add_scatter(x=ev["k"], y=ev["Silhouette"], name="Silhouette", mode="lines+markers",
                    line=dict(color="#f5b700", width=3), secondary_y=True)
    fig.update_xaxes(title="k (clusters)", dtick=1)
    fig.update_yaxes(title_text="WCSS", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette", secondary_y=True, showgrid=False)
    return _style(fig, "Choosing K: elbow + silhouette", 380)


def segment_bubbles(p, colors=CLUSTER_COLORS):
    p = p.reset_index()
    fig = px.scatter(p, x="Recency_median", y="Monetary_median", size="Customers", color=p.columns[0],
                     color_discrete_map=colors, size_max=70, log_y=True,
                     hover_data={"Frequency_median": True, "Revenue_share_%": True, "Customers": True})
    fig.update_xaxes(title="Median recency (days since last order)", autorange="reversed")
    fig.update_yaxes(title="Median net revenue £ (log)")
    return _style(fig, "Segment map — right/up is better (recency axis reversed)", 420)
