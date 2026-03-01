# Week 2: Data Preprocessing Pipeline
# One-Hot Encoding, Standard Scaling, SMOTE
# CIS 6372 – Trusted Explainability Framework for IDS

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from utils import load_nsl_kdd, convert_to_binary_labels, encode_features

# -----------------------------
# Single preprocessing pipeline (reused by training, SHAP/LIME, validation, app)
# -----------------------------

def preprocess(train_df, test_df):
    """
    Canonical preprocessing: encode (utils.encode_features), scale, SMOTE on train.
    Returns (X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler, feature_names).
    """
    # Encode using single source of truth (train defines feature set, test aligned)
    X_train = encode_features(train_df)
    feature_names = X_train.columns.tolist()
    X_test = encode_features(test_df, feature_names=feature_names)
    y_train = train_df["label"]
    y_test = test_df["label"]

    # Feature scaling: fit on train only, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE only on training data (do not resample test)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_scaled, y_train
    )

    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler, feature_names

# -----------------------------
# Step 4: Main execution
# -----------------------------

if __name__ == "__main__":

    TRAIN_PATH = "data/raw/KDDTrain+.txt"
    TEST_PATH = "data/raw/KDDTest+.txt"

    print("Loading NSL-KDD train and test sets...")
    train_df = load_nsl_kdd(TRAIN_PATH)
    test_df = load_nsl_kdd(TEST_PATH)
    train_df = convert_to_binary_labels(train_df)
    test_df = convert_to_binary_labels(test_df)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train class distribution:")
    print(train_df["label"].value_counts())

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        train_df, test_df
    )

    print("\nAfter preprocessing:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution (after SMOTE):")
    print(pd.Series(y_train).value_counts())

    print("\nTotal features after one-hot encoding:", len(feature_names))
