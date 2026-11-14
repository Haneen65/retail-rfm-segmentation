# Online Retail — Customer Analytics & Segmentation

Capstone project: cleaning → EDA → RFM → K-Means / Hierarchical / DBSCAN → Streamlit dashboard.

```
data/        raw parquet cache (+ processed/ tables written by the pipeline)
notebooks/   01_cleaning · 02_eda · 03_rfm · 04_clustering   (executed, Plotly outputs)
src/         cleaning.py · features.py · segmentation.py · viz.py · pipeline.py
reports/     RFM_Summary_Report.pdf · report_offline.html (interactive, no internet needed)
app.py       Streamlit dashboard
```

## Reproduce
```bash
pip install -r requirements.txt
python -m src.pipeline            # raw -> data/processed/*  (use: python -m src.pipeline path/to/Online_Retail.xlsx)
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub → share.streamlit.io → New app → main file `app.py`. `data/processed/` is committed, so no raw file is needed.

## Key decisions
* Returns (InvoiceNo `C…`) are **kept and netted**, never dropped; 85% matched to an earlier purchase.
* Guests (no CustomerID) form whole invoices → kept for sales analytics, excluded from customer RFM.
* Product rankings are **net of returns** (two top "bestsellers" were fully cancelled orders).
* RFM: quintiles via `qcut(rank(method="first"))`, R inverted (5 = most recent).
* Clustering on `log1p` + `StandardScaler`, K=4 (elbow kink, silhouette ≈0.34, bootstrap ARI ≈0.94, actionability).
* DBSCAN is used to flag whales/outliers, not to segment.
