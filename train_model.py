"""
Trains a Logistic Regression churn model on telco_churn.csv (the
Kaggle "Telco Customer Churn" / WA_Fn-UseC_-Telco-Customer-Churn
dataset) and saves a single pipeline (preprocessing + model) plus
metadata for the Streamlit app to load.

Run this once (and again any time you change the dataset):
    python train_model.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "telco_churn.csv"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Churn"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # TotalCharges is stored as text and has a few blank values for
    # brand-new (tenure=0) customers -- coerce to numeric and fill.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC : {auc:.3f}")
    print(classification_report(y_test, y_pred))

    # Feature names after one-hot encoding, paired with coefficients,
    # so the app can show a simple "why" explanation.
    feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    coefs = pipe.named_steps["model"].coef_[0].tolist()
    coefficients = dict(zip(feature_names, coefs))

    # Value options for each categorical dropdown, and numeric ranges,
    # so app.py can build its form directly from the data.
    cat_options = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    numeric_ranges = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())}
        for c in NUMERIC_FEATURES
    }

    joblib.dump(pipe, "model_pipeline.pkl")
    with open("model_meta.json", "w") as f:
        json.dump(
            {
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "categorical_options": cat_options,
                "numeric_ranges": numeric_ranges,
                "coefficients": coefficients,
                "accuracy": acc,
                "roc_auc": auc,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "churn_rate": float(y.mean()),
            },
            f,
            indent=2,
        )
    print("\nSaved model_pipeline.pkl, model_meta.json — ready for app.py")


if __name__ == "__main__":
    main()
