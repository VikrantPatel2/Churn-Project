import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Intelligence & Churn Dashboard")
st.caption("Churn analytics, predictive risk, and retention insights")

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/customer_churn_optimized_pipeline.joblib")
CONFIG_PATH = Path("models/customer_churn_config.joblib")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Dataset or saved model not found. Check your project folders.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df = load_data(str(DATA_PATH))
model = load_model(str(MODEL_PATH))

config = joblib.load(CONFIG_PATH) if CONFIG_PATH.exists() else {}
threshold = float(config.get("best_threshold", 0.5))

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df.loc[
    df["TotalCharges"].isna() & (df["tenure"] == 0),
    "TotalCharges"
] = 0

df = df.drop_duplicates(subset=["customerID"]).copy()

feature_columns = list(
    model.named_steps["preprocessor"].feature_names_in_
)
X = df.drop(
    columns=["Churn", "customerID"], errors="ignore"
)[feature_columns]

df["Churn_Probability"] = model.predict_proba(X)[:, 1]
df["Predicted_Churn"] = (
    df["Churn_Probability"] >= threshold
).astype(int)

df["Risk_Segment"] = pd.cut(
    df["Churn_Probability"],
    bins=[-np.inf, 0.40, 0.70, np.inf],
    labels=["Low", "Medium", "High"],
    right=False
)

total = len(df)
actual_rate = (
    df["Churn"].eq("Yes").mean() * 100
    if "Churn" in df.columns else 0
)
predicted_rate = df["Predicted_Churn"].mean() * 100
high_risk = int(df["Risk_Segment"].eq("High").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Actual Churn Rate", f"{actual_rate:.2f}%")
c3.metric("Predicted Churn Rate", f"{predicted_rate:.2f}%")
c4.metric("High-Risk Customers", f"{high_risk:,}")

st.subheader("Churn by Contract Type")
contract_summary = (
    df.groupby("Contract", observed=True)
    .agg(
        Customers=("customerID", "count"),
        Churn_Rate=(
            "Churn",
            lambda s: s.eq("Yes").mean() * 100
        )
    )
    .sort_values("Churn_Rate", ascending=False)
)
st.bar_chart(contract_summary["Churn_Rate"])
st.dataframe(contract_summary.round(2), use_container_width=True)

st.subheader("Predicted Risk Segments")
risk_summary = (
    df.groupby("Risk_Segment", observed=False)
    .agg(
        Customers=("customerID", "count"),
        Average_Churn_Probability=("Churn_Probability", "mean"),
        Average_Monthly_Charges=("MonthlyCharges", "mean")
    )
)
st.dataframe(risk_summary.round(3), use_container_width=True)

st.subheader("Find Customers by Risk")
selected_risk = st.selectbox(
    "Select risk segment",
    ["All", "High", "Medium", "Low"]
)

filtered = df.copy()
if selected_risk != "All":
    filtered = filtered[
        filtered["Risk_Segment"].astype(str) == selected_risk
    ]

display_columns = [
    "customerID", "Contract", "tenure", "MonthlyCharges",
    "Churn_Probability", "Predicted_Churn", "Risk_Segment"
]
st.dataframe(
    filtered[display_columns].sort_values(
        "Churn_Probability", ascending=False
    ),
    use_container_width=True
)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered customer data",
    data=csv,
    file_name="customer_churn_dashboard.csv",
    mime="text/csv"
)

st.info(
    "Risk bands are business-defined. Model probabilities are estimates, "
    "and retention actions should be tested against real outcomes."
)
