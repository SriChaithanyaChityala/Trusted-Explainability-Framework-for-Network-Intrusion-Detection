# Week 5: Trust Validation Framework
# CIS 6372 – Trusted Explainability Framework for IDS

import pandas as pd
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

from utils import load_data, compute_trust_score

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    DATA_PATH = "data/raw/KDDTrain+.txt"
    MODEL_PATH = "models/xgboost_ids_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    FEATURE_PATH = "models/feature_names.pkl"

    print("Loading model, scaler, and feature names...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    saved_feature_names = joblib.load(FEATURE_PATH)

    print("Loading dataset (aligned to model feature set)...")
    X, y = load_data(DATA_PATH, feature_names=saved_feature_names)

    X_unscaled = X.values
    X_scaled = scaler.transform(X)

    # SHAP
    shap_explainer = shap.TreeExplainer(model)

    # LIME
    lime_explainer = LimeTabularExplainer(
        training_data=X_unscaled,
        feature_names=X.columns.tolist(),
        class_names=["Normal", "Attack"],
        discretize_continuous=False,
        mode="classification"
    )

    def model_predict_proba(unscaled_input):
        df = pd.DataFrame(unscaled_input, columns=X.columns)
        scaled_input = scaler.transform(df)
        return model.predict_proba(scaled_input)

    # Select 20 samples for trust analysis
    sample_idx = np.random.choice(len(X_scaled), 20, replace=False)

    trust_results = []

    print("\n--- Trust Evaluation ---")

    for idx in sample_idx:

        x_scaled = X_scaled[idx].reshape(1, -1)
        x_unscaled = X_unscaled[idx]

        pred_class = model.predict(x_scaled)[0]

        # SHAP Top 5
        shap_values = shap_explainer.shap_values(x_scaled)[0]
        shap_top5 = sorted(
            zip(X.columns, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        shap_features = [f for f, _ in shap_top5]

        # LIME Top 5
        exp = lime_explainer.explain_instance(
            x_unscaled,
            model_predict_proba,
            num_features=5,
            top_labels=2
        )
        lime_features = [f.split()[0] for f, _ in exp.as_list(label=pred_class)]

        # Compute Trust (unified definition from utils)
        trust_score_pct, overlap_set, trust_level, _ = compute_trust_score(
            shap_features, lime_features
        )
        overlap_count = len(overlap_set)
        trust_score = trust_score_pct
        overlap = overlap_set

        trust_results.append({
            "Prediction": pred_class,
            "Overlap_Count": overlap_count,
            "Trust_Score (%)": trust_score,
            "Trust_Level": trust_level
        })

        print(f"\nPrediction: {pred_class}")
        print("SHAP Top5:", shap_features)
        print("LIME Top5:", lime_features)
        print("Overlap:", overlap)
        print("Trust Score:", trust_score)
        print("Trust Level:", trust_level)

    # Summary
    results_df = pd.DataFrame(trust_results)

    print("\n--- Trust Distribution Summary ---")
    print(results_df["Trust_Level"].value_counts())