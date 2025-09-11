#!/usr/bin/env python3
"""
retail_project_light.py
Lightweight pipeline for small machines:
- Loads original CSVs (data/...)
- Filters to top stores and top departments
- Keeps last `weeks` weeks
- Creates small lag features (lag1, lag2, roll4)
- Downcasts numerics to reduce memory
- Trains a small Linear Regression baseline (fast)
- Saves: data/sales_light.parquet and models/baseline_light.pkl and small figures
Usage:
    python retail_project_light.py
Tune parameters at top of file if needed.
"""

import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

# CONFIG (change here if needed)
DATA_DIR = Path("./data")
FIG_DIR = Path("./figures")
MODELS_DIR = Path("./models")

# Filenames (unchanged)
SALES_CSV = DATA_DIR / "sales data-set.csv"
STORES_CSV = DATA_DIR / "stores data-set.csv"
FEATURES_CSV = DATA_DIR / "Features data set.csv"

# Light-sampling parameters (small machine friendly)
TOP_STORES = 8       # keep only top 8 stores by revenue
TOP_DEPTS = 10       # keep only top 10 departments by revenue
WEEKS = 52           # last 52 weeks to keep
LAGS = [1, 2]        # small number of lags
ROLL_WINDOWS = [4]   # 4-week rolling mean only
USE_XGBOOST = False  # set True only if you want slightly better model and have xgboost installed
XGB_PARAMS = {"n_estimators": 50, "verbosity": 0, "random_state": 42}

# Ensure dirs
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Utility
def savefig(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved figure:", out)

def downcast(df):
    # downcast floats and ints to conserve memory
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns
    for c in float_cols:
        try:
            df[c] = pd.to_numeric(df[c], downcast="float")
        except Exception:
            pass
    for c in int_cols:
        try:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        except Exception:
            pass
    return df

def load_inputs():
    if not SALES_CSV.exists() or not STORES_CSV.exists() or not FEATURES_CSV.exists():
        print("Missing one of the input CSV files in ./data. Please put sales, stores, Features files there.")
        sys.exit(1)
    sales = pd.read_csv(SALES_CSV)
    stores = pd.read_csv(STORES_CSV)
    features = pd.read_csv(FEATURES_CSV)
    # parse Date column
    if "Date" in sales.columns:
        sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    if "Date" in features.columns:
        features["Date"] = pd.to_datetime(features["Date"], errors="coerce")
    return sales, stores, features

def sample_reduce(sales, stores, features, top_stores=TOP_STORES, top_depts=TOP_DEPTS, weeks=WEEKS):
    print("Sampling & reducing dataset to be lightweight...")
    # compute revenue per store to pick top stores
    if "Weekly_Sales" not in sales.columns:
        raise KeyError("Weekly_Sales column missing in sales CSV")
    store_rev = sales.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
    top_store_list = store_rev.head(top_stores).index.tolist()
    print("Top stores selected:", top_store_list)

    # filter sales to top stores
    sales_small = sales[sales["Store"].isin(top_store_list)].copy()

    # pick top depts (by revenue) within those stores
    dept_rev = sales_small.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False)
    top_dept_list = dept_rev.head(top_depts).index.tolist()
    print("Top departments selected:", top_dept_list)
    sales_small = sales_small[sales_small["Dept"].isin(top_dept_list)].copy()

    # filter by last `weeks`
    max_date = sales_small["Date"].max()
    cutoff = max_date - pd.Timedelta(weeks=weeks)
    sales_small = sales_small[sales_small["Date"] >= cutoff].copy()
    print(f"Keeping data from {cutoff.date()} to {max_date.date()} -> rows: {len(sales_small)}")

    # merge store meta and features (left join)
    sales_small = sales_small.merge(stores, on="Store", how="left")
    # safe merge features on Store+Date (don't require IsHoliday matching)
    if {"Store","Date"}.issubset(features.columns):
        sales_small = sales_small.merge(features, on=["Store","Date"], how="left", suffixes=("","_feat"))
    else:
        print("Features file doesn't have Store+Date - skipping merge of features.")

    # fill some exogenous with medians (fast)
    for col in ["CPI","Unemployment","Fuel_Price","Temperature"]:
        if col in sales_small.columns:
            sales_small[col] = sales_small[col].fillna(sales_small[col].median())

    # downcast
    sales_small = downcast(sales_small)
    return sales_small

def create_small_features(df, lags=LAGS, rolls=ROLL_WINDOWS):
    df = df.sort_values(["Store","Dept","Date"]).copy()
    df["Weekly_Sales"] = df["Weekly_Sales"].astype(float)
    # create small lags
    for l in lags:
        df[f"lag_{l}"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(l)
    for w in rolls:
        df[f"rollmean_{w}"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(1).rolling(window=w, min_periods=1).mean().reset_index(0,drop=True)
    # drop rows with missing smallest lag
    df = df.dropna(subset=[f"lag_{min(lags)}"]).reset_index(drop=True)
    # downcast again
    df = downcast(df)
    return df

def train_small_model(df):
    # Prepare features: only small set
    lag_feats = [c for c in df.columns if str(c).startswith("lag_")]
    roll_feats = [c for c in df.columns if str(c).startswith("rollmean_")]
    exog = [c for c in ["Temperature","Fuel_Price","CPI","Unemployment","IsHoliday"] if c in df.columns]
    X_cols = lag_feats + roll_feats + exog
    if not X_cols:
        raise ValueError("No features available for modeling. Check columns after feature engineering.")
    # Use only top frequent pairs to keep train size tiny
    freq = df.groupby(["Store","Dept"]).size().sort_values(ascending=False)
    top_pairs = freq.head(20).index.tolist()  # limit to 20 pairs only
    # Filter subset via inner merge (memory safe)
    tp_df = pd.DataFrame(top_pairs, columns=["Store","Dept"])
    subset = df.merge(tp_df, on=["Store","Dept"], how="inner")
    subset = subset.sort_values("Date").reset_index(drop=True)
    # drop rows with NaNs in X_cols
    subset = subset.dropna(subset=X_cols + ["Weekly_Sales"])
    if subset.empty:
        raise ValueError("Subset for training is empty after dropna. Reduce filters or check data.")
    # train/test split by time (last 8 weeks for test)
    max_date = subset["Date"].max()
    cutoff = max_date - pd.Timedelta(weeks=8)
    train = subset[subset["Date"] < cutoff]
    test = subset[subset["Date"] >= cutoff]
    if len(train) < 10 or len(test) < 1:
        # fallback to 6-weeks
        cutoff = max_date - pd.Timedelta(weeks=6)
        train = subset[subset["Date"] < cutoff]
        test = subset[subset["Date"] >= cutoff]
    # dropna again
    train = train.dropna(subset=X_cols); test = test.dropna(subset=X_cols)
    print("Train rows:", len(train), "Test rows:", len(test))
    model = LinearRegression()
    model.fit(train[X_cols], train["Weekly_Sales"])
    preds = model.predict(test[X_cols])
    mae = mean_absolute_error(test["Weekly_Sales"], preds)
    mape = np.mean(np.abs((test["Weekly_Sales"] - preds) / test["Weekly_Sales"])) * 100
    print(f"Light baseline model MAE={mae:.2f}, MAPE={mape:.2f}%")
    # save model and sample preds
    joblib.dump(model, MODELS_DIR / "baseline_light.pkl")
    sample_out = test[["Store","Dept","Date","Weekly_Sales"]].copy()
    sample_out["pred_lr"] = preds
    sample_out.to_csv(MODELS_DIR / "baseline_light_predictions.csv", index=False)
    print("Saved light model and predictions.")
    # quick plot (small)
    sample_pair = test.groupby(["Store","Dept"]).size().sort_values(ascending=False).index[0]
    sub = test[(test["Store"]==sample_pair[0]) & (test["Dept"]==sample_pair[1])]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(sub["Date"], sub["Weekly_Sales"], label="actual", marker="o")
        try:
            ax.plot(sub["Date"], model.predict(sub[X_cols]), label="pred", marker="o")
        except Exception:
            pass
        ax.set_title(f"Store {sample_pair[0]} Dept {sample_pair[1]} (sample)")
        ax.legend()
        savefig(fig, "baseline_light_actual_vs_pred.png")
    return model

def main():
    print("Running lightweight pipeline...")
    sales, stores, features = load_inputs()
    small = sample_reduce(sales, stores, features)
    small_fe = create_small_features(small)
    # Save small parquet
    out_parquet = DATA_DIR / "sales_light.parquet"
    small_fe.to_parquet(out_parquet, index=False)
    print("Saved reduced dataset to", out_parquet)
    # Train small model
    model = train_small_model(small_fe)
    print("Lightweight pipeline completed successfully.")

if __name__ == "__main__":
    main()
