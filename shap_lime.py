# Week 4: SHAP & LIME Explanation Generation
# CIS 6372 – Trusted Explainability Framework for IDS

import pandas as pd
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBClassifier

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
    model: XGBClassifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    saved_feature_names = joblib.load(FEATURE_PATH)

    print("Loading data (aligned to model feature set)...")
    X, y = load_data(DATA_PATH, feature_names=saved_feature_names)

    # Keep BOTH versions
    X_unscaled = X.values
    X_scaled = scaler.transform(X)

    # -----------------------------
    # SHAP (uses scaled data)
    # -----------------------------

    print("Initializing SHAP (TreeExplainer — optimal for XGBoost)...")
    shap_explainer = shap.TreeExplainer(model)

    # For final report: use 100 samples and filter to Attack (y==1) to study disagreement on malicious traffic.
    N_SAMPLES = 10
    sample_idx = np.random.choice(len(X_scaled), min(N_SAMPLES, len(X_scaled)), replace=False)
    X_scaled_sample = X_scaled[sample_idx]
    X_unscaled_sample = X_unscaled[sample_idx]

    shap_values = shap_explainer.shap_values(X_scaled_sample)

    # -----------------------------
    # LIME (uses UN-SCALED data)
    # -----------------------------

    print("Initializing LIME...")
    lime_explainer = LimeTabularExplainer(
        training_data=X_unscaled,
        feature_names=X.columns.tolist(),
        class_names=["Normal", "Attack"],
        discretize_continuous=False,  # avoid numerical issues that cause NaN
        mode="classification"
    )

    # Wrapper so LIME can call the model correctly (DataFrame keeps feature names, avoids sklearn warning)
    def model_predict_proba(unscaled_input):
        df = pd.DataFrame(unscaled_input, columns=X.columns)
        scaled_input = scaler.transform(df)
        return model.predict_proba(scaled_input)

    # -----------------------------
    # Display explanations + consensus (validation framework)
    # -----------------------------

    for i in range(len(X_scaled_sample)):
        print(f"\nSample {i+1}")

        pred_class = model.predict(X_scaled_sample[i].reshape(1, -1))[0]
        print("Predicted class:", pred_class)

        print("Top SHAP features:")
        shap_feats = sorted(
            zip(X.columns, shap_values[i]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        shap_top_names = [f for f, _ in shap_feats]

        for f, v in shap_feats:
            print(f"  {f}: {v:.4f}")

        print("Top LIME features:")
        exp = lime_explainer.explain_instance(
            X_unscaled_sample[i],
            model_predict_proba,
            num_features=5,
            labels=[pred_class],
        )
        lime_list = exp.as_list(label=pred_class)
        lime_top_names = [f for f, _ in lime_list]
        for f, w in lime_list:
            print(f"  {f}: {w:.4f}")

        # Consensus layer: Trust Score = |overlap|/5 × 100, HIGH/MEDIUM/LOW (from utils)
        trust_score_pct, overlap_set, trust_level, flag_low = compute_trust_score(
            shap_top_names, lime_top_names, top_k=5
        )
        print("Trust Score:", f"{trust_score_pct:.1f}%")
        print("Trust Level:", trust_level)
        print("Overlap:", overlap_set)
        if flag_low:
            print("*** LOW — flag for manual review ***")
