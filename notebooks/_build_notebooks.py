"""Builds the four numbered notebooks (so they are version-controlled as plain Python)."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SETUP = '''import sys, warnings; sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
from src import cleaning as cl, features as ft, segmentation as sg, viz
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_columns = 40'''


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, HERE / name)


md = lambda t: ("md", t)
code = lambda t: ("code", t)

# =====================================================================================
nb([
md("""# 01 · Business Understanding, Data Audit & Cleaning
*Sections 1–5 of the project: business understanding → dataset documentation → audit → cleaning decisions → feature engineering.*

## 1 · Business understanding
**The company.** A UK-based online giftware retailer. Its customers are a mix of **individual shoppers** and **wholesale / trade buyers** (many invoices contain 100+ units).
**The data.** Every transaction line between **1 Dec 2010 and 9 Dec 2011**.

**Decisions this analysis should support**
1. Where does the money come from — which products, customers, countries and moments?
2. How much revenue leaks through returns, and where?
3. Who should get loyalty perks, who needs a win-back offer, and who just needs a second purchase?

> *Guiding rule:* a cleaning step is only acceptable if we can say what it does to revenue."""),
code(SETUP),
md("""## 2 · Dataset loading & documentation
| Column | Meaning | Gotcha |
|---|---|---|
| `InvoiceNo` | 6-digit invoice id | Prefix **C** = cancellation (return), prefix **A** = accounting adjustment. One invoice = many lines. |
| `StockCode` | product id | Contains non-products: `POST`, `DOT`, `M`, `BANK CHARGES`, `AMAZONFEE`… |
| `Description` | product name | 1,454 missing, mostly the zero-price rows |
| `Quantity` | units on the line | negative = return / write-off |
| `InvoiceDate` | timestamp | |
| `UnitPrice` | £ per unit | 0 = giveaway/error, negative = bad-debt adjustment |
| `CustomerID` | customer id | **25% missing** → guest checkouts. It is an *identifier*, never a numeric feature |
| `Country` | customer country | |"""),
code("""raw = cl.load_raw("../data/online_retail_raw.parquet")
print(raw.shape)
raw.head()"""),
md("## 3 · Data audit"),
code("cl.audit(raw)"),
md("""**Reading the audit**
* 5,268 exact duplicate lines → remove (they would double count revenue).
* 9,288 cancellation lines, but **1,336 negative-quantity lines are *not* cancellations** (damages, stock write-offs, almost all price = 0 and guest) → they must not be treated as customer returns.
* 135,080 lines (25%) have no `CustomerID`. The guideline says *don't blindly drop* — so first we check how the gaps are distributed:"""),
code("""g = cl.guest_checkout_check(raw); g"""),
md("""**Finding:** every invoice is either *fully* identified or *fully* anonymous (0 mixed invoices). So missing IDs are genuine **guest-checkout blocks** (3,710 invoices), not random data loss.
Decision → keep them for sales/product/time analytics (they are real revenue), exclude them only from customer-level RFM, where an identity is required."""),
md("## 4 · Cleaning & data-quality decisions (with revenue impact)"),
code("""df, log = cl.clean(raw)
log.style.format({"Rows affected": "{:,}", "Revenue impact (GBP)": "£{:,.0f}"}).hide(axis="index")"""),
md("""**Why flags instead of deletes?** Every questionable line stays in the table with a boolean flag (`IsCancellation`, `IsZeroPrice`, `IsNonProduct`, `IsGuest`…). Each analysis then states exactly which flag combination it uses — nothing silently disappears, and the analysis stays reproducible.

### Returns: matched to originals, then netted
Guideline: *match cancellations with original invoices where possible.* Rule: same customer + same product, an earlier purchase exists, and the quantity bought ≥ quantity returned."""),
code("""m = cl.match_returns(df)
print(f"Cancellation lines with a customer ID: {len(m):,}")
print(f"Matched to an earlier purchase of the same product by the same customer: {m['Matched'].mean():.1%}")
m.loc[~m["Matched"], ["InvoiceNo","StockCode","Description","Quantity","CustomerID"]].head(8)"""),
md("""**85% of returns trace back to a real earlier purchase.** The remainder are mostly returns of items bought *before* 1 Dec 2010 (outside the window) — a data-window artefact, not an error. Returns are **kept and netted** (negative revenue), never deleted.

> ⚠️ **Phantom bestsellers.** Customer 16446 bought and cancelled 80,995 units of *Paper Craft, Little Birdie* (£168k) and another buyer did the same with 74,215 *Medium Ceramic Top Storage Jars*. On gross sales they are the #1 products by units. **Net of returns they vanish.** All product rankings in this project are therefore net."""),
code("""pairs = df[df["StockCode"].isin(["23166","23843"])].sort_values("InvoiceDate")
pairs[["InvoiceNo","InvoiceDate","Description","Quantity","UnitPrice","CustomerID"]]"""),
md("## 5 · Feature engineering"),
code("""# time features were created inside cl.clean(): Year, Month, YearMonth, DayOfWeek, Hour
# plus: TotalLineRevenue = Quantity x UnitPrice  and  Is_UK  (international vs home-market flag)
df[["InvoiceNo","InvoiceDate","Month","DayOfWeek","Hour","Quantity","UnitPrice","TotalLineRevenue","Is_UK","IsGuest"]].sample(6, random_state=1)"""),
code("""orders = cl.order_table(df)      # ONE row per invoice - a line is not an order
print(f"{len(df):,} transaction lines  ->  {len(orders):,} genuine product orders")
orders.head()"""),
md("""### Output of this notebook
`data/processed/lines_clean.parquet`, `orders.parquet`, `cleaning_log.csv` are produced by `python -m src.pipeline` (same code, reproducible end-to-end)."""),
code("""import subprocess, sys
subprocess.run([sys.executable, "-m", "src.pipeline"], cwd="..", check=True)
print("processed tables refreshed")"""),
], "01_cleaning.ipynb")

# =====================================================================================
nb([
md("""# 02 · Strategic EDA — from charts to business narrative
*Section 6–7. Every chart answers an executive question and ends with a **so-what**.*"""),
code(SETUP + """
df = pd.read_parquet("../data/processed/lines_clean.parquet")
orders = pd.read_parquet("../data/processed/orders.parquet")
sales = df[df.IsProductSale]
print(f"Net revenue (excl. accounting adjustments & zero-price): £{df[~df.IsAdjustment & ~df.IsZeroPrice].TotalLineRevenue.sum():,.0f}")
print(f"Gross product sales: £{sales.TotalLineRevenue.sum():,.0f}   |   orders: {len(orders):,}   |   customers: {sales.CustomerID.nunique():,}")"""),
md("## Time · how does revenue change, and when is the shop strongest?"),
code("viz.monthly_revenue(df)"),
md("""**So-what.** Revenue is flat at roughly £0.5–0.75M/month until August, then **climbs Sep → Oct → Nov (£1.02M → £1.07M → £1.46M)**. November is ~2.2× the Mar–Aug average. December 2011 looks low only because the data stops on the 9th. Gift retailers buy stock early: the surge starts in *September*, so inventory and ad budgets must be committed by August."""),
code("viz.hour_weekday_heatmap(orders)"),
code("viz.weekday_bar(orders)"),
md("""**So-what.** Trading happens **10:00–15:00** (peak at 12:00), ~84% of order value is booked between 09:00 and 16:00. **There is no Saturday trading at all**, and Sunday is the weakest day, Thursday the strongest (£2.13M). Scheduled maintenance belongs in the evening or on Saturday; paid-media bids should be raised for Tue/Thu late morning — the B2B buyer's working hours, which is itself evidence of trade customers."""),
md("## Products · what sells, and how concentrated is it?"),
code("viz.top_products(df, 'Revenue')"),
code("viz.top_products(df, 'Units')"),
code("""p = viz.pareto(df); print(p.layout.meta); p"""),
md("""**So-what.** Net of returns the revenue leader is the **Regency Cakestand 3 Tier (£164k)**, while the unit leaders are cheap items (World War 2 Gliders, Jumbo Bags, Popcorn Holders). **The Pareto principle holds almost exactly: ~21.5% of products (≈840 of 3,900) produce 80% of net revenue.** The long tail of ~3,000 SKUs adds complexity and stock cost for 20% of sales."""),
code("""dead = ft.dead_stock(df)
print(f"{len(dead)} products were in the top quartile of historical volume but sold ZERO or negative units in the last 90 days")
dead.head(12)"""),
md("""**So-what.** These 22 former volume sellers (e.g. *Tea Time Tea Towels*, *Paper Bunting Coloured Lace*, the *Charlie & Lola* range) stopped selling abruptly — likely **delisted or stocked-out** rather than unpopular. Ops should check whether they were deliberately discontinued; if not, it is lost revenue."""),
md("## Customers & geography"),
code("""c = pd.read_parquet("../data/processed/customers.parquet")
top = c.sort_values("Monetary", ascending=False)
share = lambda n: top.head(n).Monetary.sum() / c.Monetary.sum()
print(f"Top 10 customers = {share(10):.1%} of net revenue | top 20% of customers = {share(int(.2*len(c))):.1%}")
top[["CustomerID","Country","Frequency","Monetary","AvgBasketUnits"]].head(8)"""),
code("viz.country_chart(df)"),
code("""g = sales.groupby("Country").agg(Revenue=("TotalLineRevenue","sum"), Orders=("InvoiceNo","nunique"), Customers=("CustomerID","nunique")).sort_values("Revenue", ascending=False)
g["Rev_per_customer"] = g.Revenue / g.Customers
print(f"UK share of product revenue: {g.loc['United Kingdom','Revenue']/g.Revenue.sum():.1%}")
g.head(8)"""),
md("""**So-what.** The UK is **85% of revenue**. Internationally revenue per customer is *wildly* higher: the **Netherlands (9 customers → £284k)** and **EIRE (3 customers → £271k)** are a handful of B2B accounts, not consumer markets. Germany and France are the only international markets with real customer breadth (~90 customers each). Losing one Dutch account would hurt more than losing 100 UK consumers."""),
md("## Baskets · is there wholesale behaviour?"),
code("viz.basket_hist(orders)"),
code("""print(orders[["Revenue","Units","Lines"]].describe(percentiles=[.5,.9,.99]).round(0))
big = orders.Revenue >= 500
print(f"Orders >= £500: {big.mean():.1%} of orders but {orders[big].Revenue.sum()/orders.Revenue.sum():.1%} of revenue")
print(f"Orders with >= 100 units: {(orders.Units>=100).mean():.1%}")"""),
md("""**So-what.** The median order is **152 units / £302** — far beyond a household shopping basket — and 65% of orders contain 100+ units. Mean (£518) ≫ median (£302): a long wholesale tail. **24% of orders (≥£500) generate 66% of revenue.** This is a trade business wearing a consumer storefront, so segmentation must separate bulk buyers from small-batch buyers (guideline: protect the core revenue line)."""),
md("## Returns · how common, and who?"),
code("viz.returns_over_time(df)"),
code("viz.returns_by_product(df)"),
code("""canc_inv = df[df.IsCancellation].InvoiceNo.nunique() / df.InvoiceNo.nunique()
net_prod = df[df.IsCancellation & ~df.IsNonProduct & (df.UnitPrice>0)].TotalLineRevenue.sum()
print(f"Cancellation invoices: {canc_inv:.1%} of all invoices | returned value = {-net_prod/sales.TotalLineRevenue.sum():.1%} of gross product sales")
rc = df[df.IsCancellation & df.CustomerID.notna()].groupby("CustomerID").TotalLineRevenue.sum().nsmallest(5)
print("Customers with the largest return value:"); rc"""),
md("""**So-what.** Roughly **15% of invoices are cancellations but they only equal ~4.6% of gross sales value** — many small corrections, a few huge reversals. Returns are concentrated: customer 16446 alone cancelled £168k (one order), and a few B2B accounts account for most of the value. Return policy design should focus on a short list of accounts and SKUs rather than the whole base."""),
md("## Retention · are customers coming back?"),
code("viz.cohort_heatmap(df)"),
md("""**So-what.** The *December 2010* cohort (the oldest and largest, 884 customers) keeps roughly **32–40%** of its customers buying in every later month — clearly stickier than the 2011 cohorts, which settle at ~**17–30%**. Newer cohorts either reflect weaker acquisition quality or are simply too young; the sample cannot say which (descriptive data, no causal claim)."""),
md("""## Business insights (10+)
1. **Trade, not consumer.** Median order = 152 units; 65% of orders ≥100 units → treat the base as B2B-first.
2. **Revenue is front-loaded to Q4.** Nov ≈ 2.2× the Mar–Aug average; the ramp begins in September.
3. **A 10:00–15:00 weekday business.** No Saturday trading; Thursday strongest, Sunday weakest.
4. **Product Pareto holds:** ~21.5% of SKUs = 80% of net revenue; ~3,000 SKUs are long-tail.
5. **Gross rankings lie.** Two of the top-3 unit "bestsellers" were fully cancelled; only net-of-returns ranking is trustworthy.
6. **Top 20% of customers → ~74% of revenue; top 10 customers alone → ~17%.** Dependency risk.
7. **UK = 85% of product revenue**, but international is a few big B2B accounts (NL, EIRE) → high value, high concentration risk.
8. **Guest checkout is 14.7% of product revenue** (3,710 invoices, always fully anonymous) — real money that no CRM can retarget today.
9. **Returns:** ~15% of invoices, ~4.6% of value, highly concentrated in few accounts.
10. **22 former volume SKUs sold nothing in the last 90 days** → check for unintended stock-outs.
11. **65% of identified customers bought more than once** — but half of all customers last ordered 51+ days before the snapshot, a churn-watch signal.
12. **The oldest cohort is the stickiest** — loyalty value is created in the first months."""),
], "02_eda.ipynb")

# =====================================================================================
nb([
md("""# 03 · Customer-level features & RFM analysis
*Sections 8–10.*

## 8 · Customer-level feature engineering
| Feature | Definition | Why |
|---|---|---|
| **Recency (R)** | days from last *purchase* to snapshot (10 Dec 2011) | returns do not count as engagement |
| **Frequency (F)** | unique purchase invoices | invoices, **not lines** |
| **Monetary (M)** | **net** revenue = sales − returns | the guideline's "sum of all transaction revenues" |
| AOV, AvgBasketUnits, UniqueProducts, ReturnRate, Tenure | context for profiling | explain *why* a segment behaves as it does |

Excluded from customer features: guests (no identity), accounting adjustments, zero-price lines, non-product fees."""),
code(SETUP + """
df = pd.read_parquet("../data/processed/lines_clean.parquet")
cust = ft.customer_table(df)
print(f"{len(cust):,} identified customers | snapshot = {ft.snapshot_date(df).date()}")
print(f"{(cust.Monetary <= 1).sum()} customers have net revenue <= £1 (everything returned) -> cannot be scored")
cust[["Recency","Frequency","Monetary","AOV","AvgBasketUnits","ReturnRate"]].describe(percentiles=[.5,.9,.99]).round(1)"""),
md("""Frequency and Monetary are extremely right-skewed (mean ≫ median, max ≫ p99). This is the wholesale effect and is why **we log-transform before clustering**."""),
md("""## 9 · RFM analysis
**Scoring direction (read this before any chart):** each metric is cut into **5 quintiles**.
* **R: 5 = bought most recently** (lowest days) — the direction is inverted on purpose.
* **F: 5 = most invoices**, **M: 5 = most net revenue**.
* Score `'555'` = best on all three; `'155'` = big and frequent but gone quiet.

Technical note: `pd.qcut` fails on Frequency because ~35% of customers have exactly 1 invoice (tied edges). We use `rank(method="first")` before `qcut` so bins stay equal-sized."""),
code("""rfm = ft.score_rfm(cust)
print(f"{len(rfm):,} customers scored")
rfm[["CustomerID","Recency","Frequency","Monetary","R","F","M","RFM_Score","RFM_Segment"]].sort_values("Monetary", ascending=False).head(8)"""),
code("""(rfm.groupby("R").Recency.agg(["min","max","count"]).rename_axis("R score").T)"""),
md("""### Segment rules (derived from the business matrix, not forced)
| Segment | Rule | Strategy |
|---|---|---|
| **Champions** | R≥4, F≥4, M≥4 | loyalty perks, early access, referrals |
| **Loyal Customers** | R≥3, F≥3 (rest) | cross-sell related catalog |
| **At-Risk High-Value** | R≤2, F≥3, M≥3 | immediate win-back |
| **New Inflow** | R≥4, F≤2 | welcome sequence → 2nd purchase |
| **Hibernating / Lost** | R≤2, F≤2 | low-cost reactivation or let go |
| **Needs Attention** | everything else | monitor |

The guideline sketches Champions as exactly `555`; with 5 quintiles that would be a handful of people, so we use `≥4` on each axis. This is a **judgement call and is stated, not hidden**."""),
code("""seg = rfm.groupby("RFM_Segment").agg(Customers=("CustomerID","count"), Median_R=("Recency","median"), Median_F=("Frequency","median"),
        Median_M=("Monetary","median"), Revenue=("Monetary","sum"))
seg["Cust_%"] = 100*seg.Customers/seg.Customers.sum(); seg["Rev_%"] = 100*seg.Revenue/seg.Revenue.sum()
seg.sort_values("Rev_%", ascending=False).round(1)"""),
md("## 10 · RFM visualisation & interpretation"),
code("viz.rfm_heatmap(rfm)"),
code("viz.value_vs_size(rfm)"),
code("viz.segment_treemap(rfm)"),
md("""**Interpretation**
* **Champions are 22% of customers but 67% of net revenue** (median 7 orders, last bought ~11 days ago). They are the business.
* **At-Risk High-Value: 459 customers, £714k (8.6% of revenue)** that used to order ~3+ times and have been silent for ~4 months (median recency 118 days) — the most valuable *recoverable* money.
* **Hibernating / Lost: the largest segment by headcount (1,072, 25%) but only 5% of revenue** — do not spend premium retention budget here.
* **New Inflow is tiny (309, 1.6% of revenue)**: recent single-order customers. Converting them to a second order is the cheapest growth lever.

The RFM grid heatmap confirms the structure: spend rises steeply with *frequency*, and recency matters mostly in the low-F rows."""),
code("""rfm.to_parquet("../data/processed/customers_rfm_only.parquet")
rfm.RFM_Score.value_counts().head(10).to_frame("customers")"""),
], "03_rfm.ipynb")

# =====================================================================================
nb([
md("""# 04 · Clustering, segment profiling & recommendations
*Sections 11–16.*"""),
code(SETUP + """
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
df = pd.read_parquet("../data/processed/lines_clean.parquet")
c = ft.score_rfm(ft.customer_table(df))
print(len(c), "customers")"""),
md("""## 11 · Preprocessing for clustering
1. **Log transform** (`log1p`) to compress the wholesale skew.
2. **StandardScaler** → mean 0, variance 1, otherwise £ dwarfs days in Euclidean distance.
3. `CustomerID` is **not** a feature.

Skewness before / after:"""),
code("""raw_feats = c[sg.FEATURES]
print("skew BEFORE:", raw_feats.skew().round(2).to_dict())
print("skew AFTER :", np.log1p(raw_feats).skew().round(2).to_dict())
Xs, scaler = sg.preprocess(c)
print("scaled mean:", Xs.mean(0).round(3), "std:", Xs.std(0).round(3))"""),
code("""fig = go.Figure()
for f in sg.FEATURES:
    fig.add_histogram(x=np.log1p(c[f]), name=f"log {f}", opacity=.6, nbinsx=50)
fig.update_layout(barmode="overlay", title="After log1p the three distributions are usable")
viz._style(fig, "After log1p the three distributions are usable", 340)"""),
md("""## 12 · Clustering & evaluation
We don't pick K because a curve "looks right". Four independent criteria are used: **WCSS elbow, silhouette, Davies-Bouldin and bootstrap stability** (ARI between a K-Means fit on all data and fits on random 80% subsets), plus **actionability**."""),
code("""ev = sg.evaluate_k(Xs)
display(ev.round(3))
viz.elbow_silhouette(ev)"""),
code("""stab = pd.DataFrame([(k, *sg.stability(Xs, k, n_boot=10)) for k in (2,3,4,5,6)], columns=["k","ARI_mean","ARI_std"]).round(3)
stab"""),
md("""**Choosing K = 4.**
* **k=2** has the best silhouette (0.43) but only separates "active vs inactive" — it cannot drive different marketing actions.
* The elbow *kink* (largest second difference) is at k=4; WCSS flattens thereafter.
* Silhouette is practically identical for k=3 and k=4 (≈0.34) → **no separation is lost by going to 4**, and Davies-Bouldin is similar.
* **Stability:** ARI falls steadily as K grows (k=3 ≈ 0.97, **k=4 ≈ 0.94**, k=5 ≈ 0.92, k=6 ≈ 0.88). 0.94 is still very reproducible, and every extra cluster beyond 4 buys less separation and less stability.
* **Actionability:** 4 clusters map to four distinct playbooks (below).

Silhouette of 0.34 is *moderate* — customer behaviour is a continuum, not islands. We state that rather than overclaim.

### Algorithm comparison (guideline table)
**Hierarchical (Ward)** on a 1,500-customer sample: computationally heavy on all data; the dendrogram lets us verify that the tree really splits into ~4 big branches."""),
code("""import plotly.figure_factory as ff
idx = np.random.default_rng(42).choice(len(Xs), 1500, replace=False)
dend = ff.create_dendrogram(Xs[idx], orientation="bottom", linkagefun=lambda x: linkage(x, "ward"),
                            color_threshold=18)
dend.update_xaxes(showticklabels=False)
viz._style(dend, "Ward dendrogram (1,500-customer sample)", 420)"""),
code("""hc = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Xs[idx])
km_all = KMeans(4, n_init=20, random_state=42).fit(Xs)
print("ARI hierarchical vs K-Means on same sample:", round(adjusted_rand_score(hc, km_all.labels_[idx]), 3))"""),
md("""Agreement is **moderate (ARI ≈ 0.4)**, which is what we should expect: both methods minimise variance but cut a *continuous* cloud at different places. What they share is the broad structure — a large lapsed group, a small high-frequency core and an in-between recent group — so the 4-group picture is not a K-Means artefact, but the exact borders are soft.

**DBSCAN** — used as the guideline suggests: not to segment, but to **isolate the whales / anomalies**."""),
code("""lab = sg.dbscan_outliers(Xs, eps=0.4, min_samples=10)
c["DBSCAN_Outlier"] = lab == -1
print(f"DBSCAN noise points: {c.DBSCAN_Outlier.sum()} customers ({c.DBSCAN_Outlier.mean():.1%}) holding {c[c.DBSCAN_Outlier].Monetary.sum()/c.Monetary.sum():.1%} of revenue")
c.groupby("DBSCAN_Outlier")[["Recency","Frequency","Monetary","AvgBasketUnits"]].median().round(1)"""),
md("""3.8% of customers (162) are structurally unusual yet hold **37% of net revenue** — far-out bulk buyers (median basket ~300 units, median net spend ~£7.0k) who also order recently (median 16 days). These **whales are real value but would distort K-Means centroids**, so they are flagged (`DBSCAN_Outlier`, `Whale` = top 1% by revenue) and handled with account-management rather than mass marketing."""),
code("""km = KMeans(4, n_init=20, random_state=42).fit(Xs)
c["Cluster"] = km.labels_
prof = sg.profile(c, "Cluster")
names = sg.name_clusters(prof)
c["ClusterName"] = c.Cluster.map(names)
prof = sg.profile(c, "ClusterName")
prof.sort_values("Revenue_share_%", ascending=False)"""),
md("## 13 · Segment profiling"),
code("viz.radar(c)"),
code("viz.segment_bubbles(prof)"),
code("viz.galaxy(c.assign(zR=Xs[:,0], zF=Xs[:,1], zM=Xs[:,2]))"),
md("""**Names come from the statistics, not a template:**
| Cluster | Evidence |
|---|---|
| **Power Core** | ~16% of customers, **~64% of revenue**, median 10 orders, last purchase ~9 days ago, median spend £3.5k |
| **Slipping Regulars** | median 4 orders and £1.3k, but **last purchase ~52 days ago** — ~27% of customers and ~24% of revenue; the early-churn zone |
| **Fresh Explorers** | recent (≈20 days), 2 orders, £400 median — the warm second-purchase pool |
| **Lapsed One-Timers** | one order, last seen ~6 months ago (187 days), 36% of customers but **6.6% of revenue** |

Return rates and UK share are almost identical across clusters (~2%, ~90%) → **geography and returns do not explain the segments; frequency and recency do.** That is a finding, not a gap."""),
md("### Do the two segmentations agree? (rule-based RFM vs data-driven clusters)"),
code("""ct = pd.crosstab(c.RFM_Segment, c.ClusterName)
print("ARI(RFM rules, K-Means):", round(adjusted_rand_score(c.RFM_Segment, c.ClusterName), 3))
ct"""),
code("viz.sankey(c)"),
md("""Moderate agreement (ARI ≈ 0.4): rules and K-Means find the same *story* but cut it in different places. Notably the rule-based **Champions** split between *Power Core* and *Slipping Regulars* — K-Means uses raw log distance, so a frequent-but-quieter customer lands in "Slipping" even though the rules still call them Champions (R=4). **Use K-Means for marketing sizing, RFM rules for customer-level triggers** — and treat borderline customers as watch-list."""),
md("""## 14 · Business recommendations
| Segment | Size / value | Action | KPI |
|---|---|---|---|
| **Power Core** (≈16% of customers, ≈64% revenue) | protect | Dedicated account management for the top 1% whales; early access to Q4 range in **August**; no blanket discounts | retention of top-100, order frequency |
| **Slipping Regulars** (≈27% of customers, ≈24% of revenue) | **highest ROI** | Trigger a win-back when a regular passes ~45 days without ordering (their typical rhythm is shorter); bundle/volume-tier offers | % reactivated within 30 days |
| **Fresh Explorers** | grow | Welcome → 2nd-order sequence within 3 weeks; cross-sell from first basket | 2nd-order rate |
| **Lapsed One-Timers** | cheap tail | One automated reactivation e-mail, then suppress; do not spend paid media | cost per reactivation |
| **Guest checkouts** (15% of revenue) | blind spot | Incentivise account creation at checkout so they become scoreable | % identified orders |
| **Operations** | | Pre-build stock from September; maintenance on Saturday/evenings; review the 22 dead-stock SKUs; audit the few accounts behind most returns | |"""),
md("""## 15 · Limitations & next steps
* **One year of data** (and a partial December): seasonality is a single observation; can't separate trend from season.
* **Descriptive, not causal.** We can say Champions buy often; not that a perk *makes* them buy.
* **Customer 16446-type events** (huge order + full cancellation) are rare but swing totals; handled via net revenue and outlier flags.
* **25% of lines are anonymous**, so customer-level numbers describe the *identifiable* ~85% of product revenue.
* Silhouette 0.34 = overlapping, continuous segments; the labels are tools, not natural kinds.
* **Next:** (1) CLV model (BG/NBD + Gamma-Gamma) to replace the recency proxy; (2) A/B test the win-back trigger; (3) product-affinity (market-basket) per segment; (4) retrain monthly with a rolling snapshot.

## 16 · Final conclusion
The business is a **trade-led giftware wholesaler with a seasonal Q4 peak**. A small **Power Core (~16% of customers) delivers ~64% of revenue**; the largest *recoverable* pool is the **Slipping Regulars (≈£2.0M, ≈24%)**, and the large Lapsed One-Timer group is cheap noise. Cleaning decisions (net-of-returns, guest handling, de-duplication) matter: they remove phantom bestsellers and give trustworthy numbers. RFM rules give transparent, customer-level triggers, K-Means gives a stable (bootstrap ARI≈0.94) four-way market map, and DBSCAN protects both from whale distortion."""),
code("""c.to_parquet("../data/processed/customers_notebook.parquet")
print("done")"""),
], "04_clustering.ipynb")
print("notebooks written")
