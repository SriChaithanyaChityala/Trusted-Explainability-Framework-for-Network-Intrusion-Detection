# Week 3: XGBoost Model Training & Evaluation
# CIS 6372 – Trusted Explainability Framework for IDS

import pandas as pd
import joblib
import os

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils import load_nsl_kdd, convert_to_binary_labels
from preprocessing import preprocess

# -----------------------------
# Train model
# -----------------------------

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    TRAIN_PATH = "data/raw/KDDTrain+.txt"
    TEST_PATH = "data/raw/KDDTest+.txt"
    MODEL_PATH = "models/xgboost_ids_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    FEATURE_PATH = "models/feature_names.pkl"

    print("Loading NSL-KDD train and test sets...")
    train_df = load_nsl_kdd(TRAIN_PATH)
    test_df = load_nsl_kdd(TEST_PATH)
    train_df = convert_to_binary_labels(train_df)
    test_df = convert_to_binary_labels(test_df)

    print("Preprocessing (align test to train, SMOTE on train only)...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        train_df, test_df
    )

    print("Training XGBoost...")
    model = train_xgboost(X_train, y_train)

    # Global feature importance (preview before local SHAP/LIME)
    print("\n--- Global feature importance (top 15) ---")
    imp = model.feature_importances_
    for feat, score in sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {feat}: {score:.4f}")

    print("\nEvaluating on provided test set (KDDTest+.txt)...")
    y_pred = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    print("\nSaved model, scaler, and feature_names to models/")
