# app_light_enhanced.py
# Enhanced lightweight Streamlit dashboard for Retail Analytics (uses sales_light.parquet)
# Keep this file next to data/ and models/ produced by retail_project_light.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import io  # noqa: F401
import math  # noqa: F401

# Optional imports (safe: only used if files exist)
try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

st.set_page_config(page_title="Retail — Enchanced Dashboard", layout="wide")
st.title("🛒 Retail — Enchanced Dashboard")

DATA_PARQUET = Path("data/sales_light.parquet")
MODEL_PKL = Path("models/baseline_light.pkl")
ASSOC_CSV = Path("models/association_rules_sparse.csv")
CLUSTERS_CSV = Path("models/store_clusters.csv")

if not DATA_PARQUET.exists():
    st.error("Reduced dataset not found. Run retail_project_light.py first to create 'data/sales_light.parquet'.")
    st.stop()

# Load data (small)
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_parquet(DATA_PARQUET)
    # ensure Date dtype
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# quick helper: list of stores and depts
stores = sorted(df["Store"].unique())
default_store = stores[0]
store_sel = st.sidebar.selectbox("Select Store", stores, index=0)

depts_for_store = sorted(df[df["Store"] == store_sel]["Dept"].unique())
if not depts_for_store:
    st.error("No departments for selected store.")
    st.stop()
dept_sel = st.sidebar.selectbox("Select Dept", depts_for_store, index=0)

# Date range filter
min_date = df["Date"].min()
max_date = df["Date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Light controls
st.sidebar.markdown("### Visualization Options")
rolling_window = st.sidebar.selectbox("Rolling mean window (weeks)", [1, 2, 4], index=2)
show_preds = st.sidebar.checkbox("Show model predictions (fast)", value=True)
top_assoc_n = st.sidebar.slider("Top association pairs to show (if available)", 0, 40, 8)

# Filtered data
filtered = df[(df["Store"] == store_sel) & (df["Dept"] == dept_sel) & (df["Date"] >= start_date) & (df["Date"] <= end_date)].sort_values("Date").reset_index(drop=True)
if filtered.empty:
    st.warning("No data for this selection. Try a wider date range or different store/dept.")
    st.stop()

# Left column: Time series + KPIs
col1, col2 = st.columns([2.5, 1])
with col1:
    st.subheader(f"Store {store_sel} — Dept {dept_sel} Sales Time Series")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(filtered["Date"], filtered["Weekly_Sales"], label="Actual", marker='o', markersize=4, linewidth=1)
    # rolling mean
    ax.plot(filtered["Date"], filtered["Weekly_Sales"].rolling(window=rolling_window).mean(), label=f"Rolling mean ({rolling_window})", linewidth=1.5)
    # overlay model prediction if available
    model_available = MODEL_PKL.exists() and show_preds
    if model_available:
        try:
            model = joblib.load(MODEL_PKL)
            # prepare X rowwise - we select only lag_ and rollmean_ and exogs present in df
            X_cols = [c for c in filtered.columns if c.startswith("lag_") or c.startswith("rollmean_")]
            exog = [c for c in ["Temperature","Fuel_Price","CPI","Unemployment","IsHoliday"] if c in filtered.columns]
            X_cols = X_cols + exog
            # drop rows missing X_cols
            preds_df = filtered.dropna(subset=X_cols)
            if not preds_df.empty:
                preds = model.predict(preds_df[X_cols])
                ax.plot(preds_df["Date"], preds, label="Model prediction", linestyle="--", linewidth=1)
        except Exception as e:
            st.info("Model could not be loaded or predicted: " + str(e))
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    ax.legend()
    ax.set_title("Actual vs Rolling Avg" + ("" if not model_available else " & Model"))
    plt.tight_layout()
    st.pyplot(fig)

    # seasonality: average by week of year (if enough data)
    st.subheader("Weekly seasonality (avg by week-of-year)")
    df_season = df[(df["Store"]==store_sel) & (df["Dept"]==dept_sel)].copy()
    df_season["wk"] = df_season["Date"].dt.isocalendar().week
    wkavg = df_season.groupby("wk")["Weekly_Sales"].mean().reindex(range(1,54), fill_value=np.nan)
    fig2, ax2 = plt.subplots(figsize=(10,2))
    ax2.bar(wkavg.index, wkavg.values)
    ax2.set_xlabel("ISO week")
    ax2.set_ylabel("Avg weekly sales")
    ax2.set_xlim(1,53)
    plt.tight_layout()
    st.pyplot(fig2)

with col2:
    st.subheader("Key KPIs (selection)")
    avg_sales = filtered["Weekly_Sales"].mean()
    max_sales = filtered["Weekly_Sales"].max()
    last_sales = filtered.sort_values("Date").iloc[-1]["Weekly_Sales"]
    st.metric("Avg weekly sales", f"{avg_sales:,.0f}")
    st.metric("Max weekly sales", f"{max_sales:,.0f}")
    st.metric("Last week's sales", f"{last_sales:,.0f}")

    # model MAPE (approx) on filtered selection if model exists
    if model_available:
        # compute MAPE for rows where we can compute X_cols
        try:
            preds_df = filtered.dropna(subset=X_cols)
            if not preds_df.empty:
                preds = model.predict(preds_df[X_cols])
                mape_val = np.mean(np.abs((preds_df["Weekly_Sales"] - preds) / preds_df["Weekly_Sales"])) * 100
                st.metric("Model MAPE (on available rows)", f"{mape_val:.2f}%")
        except Exception:
            pass

    # small download button for the filtered data
    st.subheader("Download")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name=f"store{store_sel}_dept{dept_sel}_filtered.csv", mime="text/csv")

# Middle area: Top departments and heatmap
st.markdown("---")
left, right = st.columns([1.8,1])
with left:
    st.subheader(f"Top Departments in Store {store_sel}")
    by_dept = df[df["Store"]==store_sel].groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False).head(12)
    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.bar(by_dept.index.astype(str), by_dept.values)
    ax3.set_xticklabels(by_dept.index.astype(str), rotation=45)
    ax3.set_xlabel("Dept")
    ax3.set_ylabel("Total Sales")
    plt.tight_layout()
    st.pyplot(fig3)

with right:
    st.subheader("Store × Week heatmap (sample small)")
    # pivot small sample (dept as rows, week index as columns)
    sample = df[(df["Store"]==store_sel) & (df["Date"]>= (end_date - pd.Timedelta(weeks=52)))]
    if sample.empty:
        st.write("Not enough data for heatmap.")
    else:
        # pivot Dept x week-of-year average sales
        sample["wk"] = sample["Date"].dt.isocalendar().week
        pivot = sample.pivot_table(index="Dept", columns="wk", values="Weekly_Sales", aggfunc="mean").fillna(0)
        # limit to top 10 depts in pivot
        topdept = pivot.sum(axis=1).sort_values(ascending=False).head(8).index
        small_pivot = pivot.loc[topdept]
        fig4, ax4 = plt.subplots(figsize=(8,3))
        im = ax4.imshow(small_pivot, aspect="auto")
        ax4.set_yticks(range(len(small_pivot.index)))
        ax4.set_yticklabels(small_pivot.index.astype(str))
        ax4.set_xlabel("ISO week")
        plt.colorbar(im, ax=ax4, orientation="vertical", fraction=0.04)
        st.pyplot(fig4)

# Association network (if available)
st.markdown("---")
st.subheader("Association Rules (department co-occurrence)")
if ASSOC_CSV.exists() and NX_AVAILABLE:
    assoc = pd.read_csv(ASSOC_CSV)
    if assoc.empty:
        st.write("No association rules found in the CSV.")
    else:
        topn = assoc.sort_values(["lift","support"], ascending=[False,False]).head(top_assoc_n)
        st.write(f"Showing top {len(topn)} pairs by lift")
        # draw small network
        G = nx.Graph()
        for _, r in topn.iterrows():
            a = str(r["dept_A"]); b = str(r["dept_B"])  # noqa: E702
            G.add_edge(a, b, weight=float(r["lift"]))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        fig5, ax5 = plt.subplots(figsize=(6,4))
        edges = G.edges(data=True)
        weights = [max(0.5, 2*data["weight"]) for _,_,data in edges]
        nx.draw_networkx_nodes(G, pos, node_size=700, ax=ax5)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax5)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8, ax=ax5)
        ax5.set_title("Dept association network (top pairs)")
        ax5.axis("off")
        st.pyplot(fig5)

        # show table
        st.dataframe(topn[["dept_A","dept_B","count_cooccurrence","support","confidence_A_to_B","confidence_B_to_A","lift"]].reset_index(drop=True))
else:
    if ASSOC_CSV.exists() and not NX_AVAILABLE:
        st.info("Association rules CSV exists, but `networkx` is not installed. Install networkx to show network.")
    else:
        st.info("No association rules file found. Run association stage to compute rules.")

# Clusters info (if available)
st.markdown("---")
st.subheader("Store Clusters (if available)")
if CLUSTERS_CSV.exists():
    clusters = pd.read_csv(CLUSTERS_CSV)
    if not clusters.empty:
        st.write("Store clusters loaded. Basic summary:")
        st.dataframe(clusters[["Store","cluster"]].sort_values("cluster").reset_index(drop=True))
        # quick cluster counts
        st.bar_chart(clusters["cluster"].value_counts().sort_index())
    else:
        st.write("clusters file is empty.")
else:
    st.info("No store clusters file found. (optional) Run clustering stage to create models/store_clusters.csv")

# What-if simulator (approximate & fast)
st.markdown("---")
st.subheader("What-if simulator (approximate)")
st.write("Adjust a single exogenous input (MarkDown1 or Fuel_Price) for the selected Store/Dept. This is an *approximate* effect because we do not re-simulate lag features — intended for quick demos.")

sim_col = st.selectbox("Adjust parameter", ["MarkDown1", "Fuel_Price"])
if sim_col not in df.columns:
    st.write(f"{sim_col} not in dataset for this selection.")
else:
    last = df[(df["Store"]==store_sel) & (df["Dept"]==dept_sel)].sort_values("Date").tail(1)
    if last.empty:
        st.write("No recent row to run simulation.")
    else:
        original = float(last.iloc[0].get(sim_col, 0.0))
        if sim_col == "MarkDown1":
            delta = st.slider("Absolute change in MarkDown1 (±)", -2000, 2000, 0, step=100)
            new_val = max(0.0, original + delta)
        else:
            # Fuel price slider
            delta = st.slider("Absolute change in Fuel_Price (±)", -3.0, 3.0, 0.0, step=0.1)
            new_val = max(0.0, original + delta)

        st.write(f"Original {sim_col}: {original:.2f} → New: {new_val:.2f}")

        # approximate prediction: load model and replace only exog col with new_val in last row features
        if MODEL_PKL.exists():
            try:
                model = joblib.load(MODEL_PKL)
                X_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rollmean_")]
                exog = [c for c in ["Temperature","Fuel_Price","CPI","Unemployment","IsHoliday","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"] if c in df.columns]
                X_cols = X_cols + exog
                X_row = last[X_cols].fillna(0).iloc[0].to_dict()
                # replace the sim_col value
                if sim_col in X_row:
                    X_row[sim_col] = new_val
                # prepare 1-row DataFrame
                Xdf = pd.DataFrame([X_row])
                pred_orig = None
                pred_new = None
                try:
                    pred_orig = model.predict(pd.DataFrame([last[X_cols].fillna(0).iloc[0]]))[0]
                except Exception:
                    pred_orig = None
                try:
                    pred_new = model.predict(Xdf)[0]
                except Exception:
                    pred_new = None
                st.write("### Approximate effect on predicted Weekly Sales")
                if pred_orig is not None and pred_new is not None:
                    delta_sales = pred_new - pred_orig
                    pct = (delta_sales / pred_orig * 100) if pred_orig != 0 else np.nan
                    st.metric("Predicted change (est.)", f"{delta_sales:.0f} ({pct:.2f}%)")
                    st.write(f"Predicted before: {pred_orig:.0f}  → after: {pred_new:.0f}")
                else:
                    st.write("Model prediction not available for this row.")
            except Exception as e:
                st.write("Model load/prediction error:", e)
        else:
            st.write("No model found to run simulation. Train lightweight model first.")

st.markdown("---")
st.write("Small tips: reduce Top Stores/Depts in the light pipeline if app still lags. For network graphs, install `networkx` (`pip install networkx`).")

st.caption("This dashboard is built for quick demos on low-memory machines.")
