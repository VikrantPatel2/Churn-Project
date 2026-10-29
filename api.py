
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_optimized_pipeline.joblib"
CONFIG_PATH = BASE_DIR / "models" / "customer_churn_config.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

config = (
    joblib.load(CONFIG_PATH)
    if CONFIG_PATH.exists()
    else {}
)

# Use the feature schema saved by the preprocessing pipeline.
feature_columns = list(
    model.named_steps["preprocessor"].feature_names_in_
)

# Adjust these keys to match your saved configuration, if needed.
threshold = 0.5

if isinstance(config, dict):
    for key in ["best_threshold", "optimal_threshold", "threshold"]:
        if key in config:
            threshold = float(config[key])
            break

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn risk using the trained ML pipeline.",
    version="1.0.0"
)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=1000)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(customer: Customer):
    try:
        customer_data = customer.model_dump()

        X_new = pd.DataFrame(
            [customer_data],
            columns=feature_columns
        )

        probability = float(model.predict_proba(X_new)[0, 1])
        prediction = int(probability >= threshold)

        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn_prediction": prediction,
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "risk_level": risk,
            "decision_threshold": threshold
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc
