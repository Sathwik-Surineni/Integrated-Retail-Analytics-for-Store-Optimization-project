# Integrated Retail Analytics for Store Optimization



A data analytics project that demonstrates **retail sales forecasting and store optimization** using real-world style datasets.  
The project covers the complete pipeline from **data ingestion → cleaning → feature engineering → exploratory data analysis (EDA) → predictive modeling → clustering → association rules → interactive dashboard**.


The project is about retail analytics using advanced machine learning for store optimization. It looks like the document outlines how to leverage data (likely sales, inventory, customer, and operations data) to improve decisions such as:

-Product placement & shelf optimization

-Demand forecasting

-Inventory management

-Customer behavior analysis


---

## Objective

Optimize retail store performance through integrated analytics.

Use ML models to improve decision-making in real time.

## Core Areas of Optimization

Store Layout & Shelf Optimization: Maximize sales by placing products strategically.

Demand Forecasting: Predict future demand using time-series ML models.

Customer Insights: Segment customers and personalize promotions.

Inventory Optimization: Reduce stockouts/overstock by balancing supply with demand.

Pricing Strategy: Dynamic pricing using competitor, demand, and cost data.

## Technology Stack (likely to be included)

Data Sources: POS transactions, customer loyalty cards, online data, supply chain records.

ML/AI Models: Time series (ARIMA, Prophet, LSTM), Clustering (K-Means, DBSCAN), Classification (XGBoost, Random Forest), Reinforcement learning for layout/placement optimization.

Visualization: Dashboards for decision-making (Power BI, Tableau, or Python libraries).

## Expected Outcomes

Better shelf space utilization.

Higher revenue through optimized pricing & promotions.

Reduced wastage & improved supply chain efficiency.

Personalized customer experience.


#### This project builds an integrated retail analytics pipeline for store optimization using weekly sales, store metadata, and exogenous weekly features. We perform EDA, create lag-based features, and train forecasting models to predict weekly sales at Store–Dept level. We augment forecasting with store segmentation, association-rule based basket analysis, and optimization simulations for markdowns and shelf allocation. The pipeline includes explainability components and an interactive Streamlit dashboard for business users.

## 📂 Project Structure

```
.
├── app.py # Basic Streamlit app
├── app_light_enhanced.py # Enhanced Streamlit dashboard (recommended)
├── retail_project.py # Full pipeline (heavy, full dataset)
├── retail_project_light.py # Lightweight pipeline (runs faster on laptops)
├── inspect_rules.py # Utility to inspect association rules
├── requirements.txt # Python dependencies
├── figures/ # Generated plots (PNG)
├── models/ # Trained models & outputs (small)
└── data/ # Raw & processed datasets (ignored in Git)
```

- **`figures/`** → timeseries, heatmaps, feature importances, cluster plots  
- **`models/`** → saved `.pkl` models, predictions, association rules, clusters  
- **`data/`** → raw CSVs and parquet (ignored by default for repo cleanliness)

---

## 🚀 Features

- **Data Preprocessing**
  - Merges sales, stores, and features datasets
  - Handles missing values, creates lag & rolling features
- **Exploratory Data Analysis**
  - Sales time-series
  - Top stores & departments
  - Seasonal patterns
  - Heatmaps
- **Forecasting Models**
  - Baseline: Linear Regression
  - Advanced: XGBoost (optional)
- **Clustering**
  - Store-level clustering using PCA + KMeans
- **Association Rules**
  - Department co-occurrence rules (Apriori-style, memory-safe)
- **Interactive Dashboard**
  - KPIs, rolling averages, and sales prediction overlay
  - Seasonality & heatmaps
  - Association rule network
  - What-if simulator (markdown/fuel price changes)

---

## ⚙️ Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Sathwik-Surineni/Integrated-Retail-Analytics-for-Store-Optimization-project.git
cd Integrated-Retail-Analytics-for-Store-Optimization-project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PowerShell
# or
source .venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```
## ▶️ Usage

1. Run the lightweight pipeline (recommended)

This creates a reduced dataset and a small model.
```
python retail_project_light.py
```
2. Launch the dashboard
```
streamlit run app_light_enhanced.py
```
Open the link (default: http://localhost:8501).

3. Run the full pipeline (optional, requires more RAM)
```
python retail_project.py --run_all
```
## Example Outputs

figures/total_weekly_sales_timeseries.png → total sales trend

figures/top_stores_by_sales.png → top stores by sales

figures/store_week_heatmap_sample.png → store × week heatmap

figures/xgb_feature_importances.png → XGBoost feature importances

models/store_clusters.csv → store cluster assignments

models/association_rules_sparse.csv → top association rules


## Technologies Used

Python 3.10+

Pandas, NumPy → data handling

Matplotlib → visualization

Scikit-learn → baseline ML models, clustering

XGBoost (optional) → advanced forecasting

Streamlit → interactive dashboard

mlxtend (optional) → association rules

PyArrow → parquet support

## Deliverables

Cleaned & processed dataset (sales_light.parquet)

Trained lightweight baseline model (baseline_light.pkl)

Interactive dashboard (app_light_enhanced.py)

Visualizations (saved in figures/)

Models, clusters, and association rules (saved in models/)



## drive link 

here's my google drive link so you can check it regarding thr project ;
```
https://drive.google.com/drive/folders/1AqYg2KxxNQqNP6UZqVSJ1OeR3APyT8wC?usp=sharing
```
   
