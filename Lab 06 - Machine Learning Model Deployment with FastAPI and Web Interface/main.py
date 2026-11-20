"""
Lab 06: Machine Learning Model Deployment with FastAPI
Course: Introduction to Artificial Intelligence (AI-101L)
Instructor: Dr. Muhammad Awais

This module implements a production-ready RESTful inference API for deploying
a trained clinical machine learning model (Logistic Regression for Diabetes Prediction).
"""

import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

# Try importing FastAPI; provide graceful fallback description if not installed
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None


# --- 1. Load Serialized Machine Learning Model Artifact ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mymdoel.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Train and export model in Lab 04.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Ensure feature alignment with training schema
FEATURES: List[str] = list(getattr(model, "feature_names_in_", [
    "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]))


# --- 2. Define Pydantic Request & Response Data Schemas ---
class PredictRequest(BaseModel):
    Glucose: float = Field(..., ge=0.0, le=400.0, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(..., ge=0.0, le=250.0, description="Diastolic blood pressure (mm Hg)")
    Insulin: float = Field(..., ge=0.0, le=900.0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0.0, le=80.0, description="Body Mass Index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=3.0, description="Diabetes pedigree genetic score")
    Age: float = Field(..., ge=1.0, le=130.0, description="Patient age in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Glucose": 148.0,
                "BloodPressure": 72.0,
                "Insulin": 0.0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50.0
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_label: int = Field(..., description="0 for Non-Diabetic, 1 for Diabetic")
    prediction: str = Field(..., description="Human-readable diagnostic classification")
    probability: float = Field(..., description="Estimated posterior probability of diabetes [0.0 - 1.0]")
    risk_level: str = Field(..., description="Stratified clinical risk tier")
    timestamp: str = Field(..., description="UTC ISO-8601 inference timestamp")


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_count: int
    features: List[str]
    classes: List[int]
    intercept: float
    coefficients: Dict[str, float]


# --- 3. Core Prediction Logic (Engine) ---
def make_prediction(data_dict: Dict[str, float]) -> Dict[str, Any]:
    """Execute model inference given patient feature dictionary."""
    df = pd.DataFrame([data_dict])[FEATURES]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    # Clinical risk stratification
    if prob < 0.35:
        risk = "Low Risk"
    elif prob < 0.65:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return {
        "predicted_label": pred,
        "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
        "probability": round(prob, 4),
        "risk_level": risk,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# --- 4. FastAPI Application Setup ---
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Clinical Diabetes Prediction API",
        description="Production REST microservice serving a Scikit-Learn Logistic Regression model.",
        version="2.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["General"])
    def home():
        return {
            "service": "Clinical Diabetes Prediction API",
            "status": "online",
            "version": "2.0.0",
            "docs_url": "/docs",
            "health_check": "/health",
            "model_info": "/model-info",
            "predict_endpoint": "/predict"
        }

    @app.get("/health", tags=["Monitoring"])
    def health():
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
    def model_info():
        coef_dict = {feat: float(c) for feat, c in zip(FEATURES, model.coef_[0])}
        return {
            "model_type": type(model).__name__,
            "feature_count": len(FEATURES),
            "features": FEATURES,
            "classes": [int(c) for c in model.classes_],
            "intercept": float(model.intercept_[0]),
            "coefficients": coef_dict
        }

    @app.post("/predict", response_model=PredictResponse, tags=["Inference"])
    def predict(req: PredictRequest):
        try:
            return make_prediction(req.model_dump())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

else:
    app = None
    print("[WARNING] FastAPI is not installed. Use 'pip install fastapi uvicorn' to run the web API.")


if __name__ == "__main__":
    import sys
    print("Testing local standalone inference...")
    test_patient = {
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50.0
    }
    result = make_prediction(test_patient)
    print("Inference Result on Test Patient:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if FASTAPI_AVAILABLE:
        print("\nStarting Uvicorn server on port 8000...")
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
