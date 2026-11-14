"""Builds the interactive story page.

  python reports/build_report.py            -> reports/report_offline.html (plotly.js inlined, works anywhere)
                                               reports/report_cdn.html     (plotly.js from jsDelivr, for the web artifact)
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import features as ft, segmentation as sg, viz  # noqa: E402

P = ROOT / "data" / "processed"
df = pd.read_parquet(P / "lines_clean.parquet")
c = pd.read_parquet(P / "customers.parquet")
o = pd.read_parquet(P / "orders.parquet")

# ---- numbers quoted in the page, computed here so text and charts can't drift apart
prof = sg.profile(c, "ClusterName")
pc = prof.loc["Power Core"]; sr = prof.loc["Slipping Regulars"]; fe = prof.loc["Fresh Explorers"]; lo = prof.loc["Lapsed One-Timers"]
net = df[~df.IsAdjustment & ~df.IsZeroPrice].TotalLineRevenue.sum()
sales = df[df.IsProductSale]
uk = sales[sales.Country == "United Kingdom"].TotalLineRevenue.sum() / sales.TotalLineRevenue.sum()
guest = sales[sales.IsGuest].TotalLineRevenue.sum() / sales.TotalLineRevenue.sum()
mm = df[~df.IsAdjustment & ~df.IsZeroPrice].groupby("YearMonth").TotalLineRevenue.sum()
nov_x = mm["2011-11"] / mm["2011-03":"2011-08"].mean()
par = viz.pareto(df).layout.meta
champ = c[c.RFM_Segment == "Champions"]
ar = c[c.RFM_Segment == "At-Risk High-Value"]
ev = sg.evaluate_k(sg.preprocess(c)[0])

N = dict(
    customers=len(c), net=net, orders=len(o), pc_cust=pc["Customer_share_%"], pc_rev=pc["Revenue_share_%"],
    sr_rev=sr["Revenue_share_%"], sr_money=sr["Monetary_sum"], sr_cust=sr["Customer_share_%"],
    lo_cust=lo["Customer_share_%"], lo_rev=lo["Revenue_share_%"], fe_n=int(fe["Customers"]), uk=uk * 100, guest=guest * 100,
    nov_x=nov_x, pct80=par["pct80"], n80=par["n80"], nprod=par["n_products"],
    ch_cust=100 * len(champ) / len(c), ch_rev=100 * champ.Monetary.sum() / c.Monetary.sum(),
    ar_n=len(ar), ar_money=ar.Monetary.sum(),
)

FIGS = {
    "galaxy": viz.galaxy(c), "monthly": viz.monthly_revenue(df), "heat": viz.hour_weekday_heatmap(o),
    "pareto": viz.pareto(df), "top": viz.top_products(df, "Revenue"), "country": viz.country_chart(df),
    "basket": viz.basket_hist(o), "vs": viz.value_vs_size(c, "ClusterName", viz.CLUSTER_COLORS),
    "rfm": viz.rfm_heatmap(c), "sankey": viz.sankey(c), "radar": viz.radar(c), "bubbles": viz.segment_bubbles(prof),
    "elbow": viz.elbow_silhouette(ev), "cohort": viz.cohort_heatmap(df), "ret": viz.returns_by_product(df),
}
fig_json = {k: json.loads(v.to_json()) for k, v in FIGS.items()}

CSS = """
:root{--bg:#f6f5f2;--surface:#ffffff;--ink:#16181d;--muted:#5d6472;--line:#dcdad3;--gold:#9a6b00;--gold-bg:#fbefc9;--teal:#0f766e;--red:#b4332f;--chart-bg:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b0f17;--surface:#121826;--ink:#e9ecf2;--muted:#9aa3b5;--line:#232c40;--gold:#f5b700;--gold-bg:#2a2208;--teal:#2ec4b6;--red:#ff7a76;--chart-bg:#121826;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0b0f17;--surface:#121826;--ink:#e9ecf2;--muted:#9aa3b5;--line:#232c40;--gold:#f5b700;--gold-bg:#2a2208;--teal:#2ec4b6;--red:#ff7a76;--chart-bg:#121826;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:16px/1.6 Inter,system-ui,sans-serif;padding-inline:max(16px,env(safe-area-inset-left,0px)) max(16px,env(safe-area-inset-right,0px));padding-block:0}
.wrap{max-width:1040px;margin:0 auto}
h1,h2,h3{font-family:"Space Grotesk",Inter,system-ui,sans-serif;text-wrap:balance;line-height:1.12;margin:0}
.eyebrow{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
header{padding:56px 0 24px}
h1{font-size:clamp(2.1rem,6vw,3.7rem);font-weight:700;letter-spacing:-.02em;margin:10px 0 14px}
h1 em{font-style:normal;color:var(--gold)}
.lede{font-size:1.13rem;color:var(--muted);max-width:62ch;margin:0}
.strip{display:flex;flex-wrap:wrap;gap:6px 28px;margin-top:22px;padding-top:16px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}
.strip div{min-width:0}.strip b{font-family:"Space Grotesk",sans-serif;font-size:1.35rem;display:block}
.strip span{font-size:.78rem;color:var(--muted)}
section{padding:40px 0 8px}
h2{font-size:clamp(1.5rem,3.4vw,2.1rem);margin:6px 0 10px;letter-spacing:-.01em}
p{max-width:68ch;margin:0 0 14px}
.chart{background:var(--chart-bg);border:1px solid var(--line);border-radius:14px;padding:6px;margin:14px 0;min-width:0;overflow:hidden}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}
.so{border-left:3px solid var(--gold);padding:2px 0 2px 14px;margin:6px 0 18px;max-width:68ch}
.so b{color:var(--gold)}
.warn{background:var(--gold-bg);border-radius:12px;padding:14px 18px;margin:14px 0;max-width:72ch}
table{border-collapse:collapse;width:100%;font-size:.92rem;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600}
.tw{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:14px}
.dot{display:inline-block;width:.7em;height:.7em;border-radius:50%;margin-right:.5em}
footer{padding:36px 0 56px;color:var(--muted);font-size:.85rem}
@media print{.chart{break-inside:avoid}section{break-inside:avoid-page}}
"""

def sec(eyebrow, title, body):
    return f'<section><div class="eyebrow">{eyebrow}</div><h2>{title}</h2>{body}</section>'

def chart(k):
    return f'<div class="chart"><div id="f_{k}"></div></div>'

def so(t):
    return f'<div class="so"><b>So what.</b> {t}</div>'

COL = viz.CLUSTER_COLORS
play = [
    ("Power Core", pc, "Protect", "Named account managers for the top 1%, a private preview of the Q4 range in August, no blanket discounts.", "Top-100 retention, order frequency"),
    ("Slipping Regulars", sr, "Win back first", "Trigger an offer when a regular passes about 45 days without ordering. Bundles and volume tiers beat percentage discounts.", "Reactivated within 30 days"),
    ("Fresh Explorers", fe, "Get the 2nd order", "Welcome sequence, cross-sell from the first basket, a reward that unlocks on order two within three weeks.", "Second-order rate"),
    ("Lapsed One-Timers", lo, "Send one email", "A single automated reactivation message, then suppress. No paid media.", "Cost per reactivation"),
]
rows = "".join(
    f'<tr><td><span class="dot" style="background:{COL[n]}"></span><b>{n}</b></td><td>{p["Customers"]:,.0f}<br><span style="color:var(--muted)">{p["Customer_share_%"]:.0f}% of customers</span></td>'
    f'<td>{p["Revenue_share_%"]:.1f}%<br><span style="color:var(--muted)">£{p["Monetary_sum"]/1e6:,.2f}M</span></td>'
    f'<td>{p["Frequency_median"]:.0f} orders<br><span style="color:var(--muted)">last {p["Recency_median"]:.0f} days ago · £{p["Monetary_median"]:,.0f}</span></td>'
    f'<td><b>{a}.</b> {t}<br><span style="color:var(--muted)">KPI: {k}</span></td></tr>'
    for n, p, a, t, k in play)

BODY = f"""
<header>
 <div class="eyebrow">Online Retail · 1 Dec 2010 – 9 Dec 2011 · {N['customers']:,} scored customers</div>
 <h1>{N['pc_cust']:.0f}% of customers pay <em>{N['pc_rev']:.0f}%</em> of the bills</h1>
 <p class="lede">A giftware retailer that looks like a shop and sells like a wholesaler. This page follows the money from the first cleaned transaction to four customer segments, and ends with what to do about each one.</p>
 <div class="strip">
  <div><b>£{N['net']/1e6:,.2f}M</b><span>net revenue, after returns</span></div>
  <div><b>{N['orders']:,}</b><span>product orders</span></div>
  <div><b>{N['uk']:.0f}%</b><span>of product revenue from the UK</span></div>
  <div><b>{N['guest']:.0f}%</b><span>of product revenue is guest checkout</span></div>
 </div>
</header>

<section>
 <div class="eyebrow">The customer galaxy</div>
 <h2>Every dot is a customer</h2>
 <p>Position comes from log-scaled Recency, Frequency and Monetary value. Drag to rotate, click a name in the legend to hide a group.</p>
 {chart('galaxy')}
 {so(f"Four groups emerge. The <b>Power Core</b> ({N['pc_cust']:.0f}% of customers) sits at the high-frequency, high-spend end and brings in {N['pc_rev']:.0f}% of revenue. The <b>Lapsed One-Timers</b> are {N['lo_cust']:.0f}% of customers and {N['lo_rev']:.1f}% of revenue.")}
</section>

{sec("The shop", "A weekday, daytime, Q4-heavy business", f'''
<p>Revenue is flat at roughly £0.5–0.75M a month until August, then climbs each month into November.</p>
{chart('monthly')}
{so(f"November is {N['nov_x']:.1f}× the March–August average, and the ramp starts in September. Stock and ad budgets have to be committed by August. December is low only because the data stops on the 9th.")}
{chart('heat')}
{so("Orders arrive between 10:00 and 15:00 on weekdays, with a peak at noon. There is no Saturday trading at all. These are the working hours of trade buyers, which is the first sign of wholesale behaviour. Schedule maintenance for evenings and Saturdays.")}
{chart('basket')}
{so("The median order is 152 units and £302. 65% of orders hold 100 units or more, and the 24% of orders worth £500 or more produce 66% of revenue.")}
''')}

{sec("The products", f"{N['pct80']:.0f}% of products make 80% of revenue", f'''
<p>The Pareto rule holds almost exactly: about {N['n80']:,} of {N['nprod']:,} products produce 80% of net revenue.</p>
<div class="grid2">{chart('pareto')}{chart('top')}</div>
<div class="warn"><b>Phantom bestsellers.</b> One buyer ordered and cancelled 80,995 units of <i>Paper Craft, Little Birdie</i> (£168k), and another did the same with 74,215 storage jars. On gross sales they are the top two products by units. Net of returns they disappear, so every ranking here is net.</div>
{chart('ret')}
{so("Cancellations are about 15% of invoices but only 4.6% of sales value: many small corrections and a few huge reversals. Returns policy should target a short list of accounts and products.")}
''')}

{sec("The geography", "International is a handful of big accounts", f'''
<p>The UK is {N['uk']:.0f}% of product revenue.</p>
{chart('country')}
{so("The Netherlands brings £284k from 9 customers and EIRE £271k from 3. Germany and France are the only international markets with real customer breadth, about 90 customers each. Losing one Dutch account would cost more than losing a hundred UK consumers.")}
''')}

{sec("The segments", "Two ways of cutting the same customers", f'''
<p>RFM scores (each metric in 5 quintiles, Recency inverted so 5 means most recent) give transparent rules. K-Means on log-scaled, standardised R, F and M gives a data-driven map. Silhouette is moderate at 0.34, so the borders are soft.</p>
<div class="grid2">{chart('rfm')}{chart('vs')}</div>
{so(f"RFM Champions are {N['ch_cust']:.0f}% of customers and {N['ch_rev']:.0f}% of revenue. The <b>At-Risk High-Value</b> group is {N['ar_n']} customers holding £{N['ar_money']/1e3:,.0f}k that have been silent for about four months.")}
{chart('sankey')}
{chart('radar')}
<div class="grid2">{chart('elbow')}{chart('bubbles')}</div>
<p>K = 4 was chosen on four counts: the elbow kink sits at 4, silhouette is the same as k=3, bootstrap stability stays high (ARI about 0.94), and each cluster maps to a different marketing action. K = 2 scores best on silhouette but only splits active from inactive.</p>
''')}

{sec("The playbook", "What to do with each segment", f'''
<div class="tw"><table><thead><tr><th>Segment</th><th>Size</th><th>Revenue</th><th>Behaviour (median)</th><th>Action</th></tr></thead><tbody>{rows}</tbody></table></div>
<p style="margin-top:16px">The largest recoverable pool is the Slipping Regulars: £{N['sr_money']/1e6:,.1f}M of revenue from customers who still order, only less often. Two data points sit outside every segment: a handful of whales (flagged by DBSCAN, 3.8% of customers, 37% of revenue) belong with account management, and the {N['guest']:.0f}% of revenue from guest checkout cannot be reached until those buyers create accounts.</p>
{chart('cohort')}
''')}

<footer class="wrap">
 <b>Limits.</b> One year of data, so seasonality is a single observation. Everything here is descriptive and cannot show what causes what.
 Customer-level figures cover the identifiable customers only. Reproduce it with <code>python -m src.pipeline</code>, then open the notebooks in order.
</footer>
"""

SCRIPT = """
const FIGS=%s;
const render=()=>{for(const k in FIGS){const el=document.getElementById('f_'+k);if(!el)continue;
 const f=FIGS[k];f.layout.autosize=true;if(window.innerWidth<560){f.layout.margin=Object.assign({},f.layout.margin,{l:36,r:10});f.layout.title=null;f.layout.legend=Object.assign({},f.layout.legend,{orientation:'h',y:-0.15});}
 Plotly.react(el,f.data,f.layout,{responsive:true,displaylogo:false});}};
render();
""" % json.dumps(fig_json)

FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap">'


def page(plotly_tag):
    return (f'<title>Retail Customer Galaxy</title><meta name="claude:viewport-fit-sides" content="cover">{FONTS}'
            f'<style>{CSS}</style><div class="wrap">{BODY}</div>{plotly_tag}<script>{SCRIPT}</script>')


out = ROOT / "reports"
pjs = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
(out / "report_cdn.html").write_text(page('<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"></script>'), encoding="utf8")
(out / "report_offline.html").write_text(
    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    + page(f"<script>{pjs.read_text(encoding='utf8')}</script>"), encoding="utf8")
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in N.items()})
