# Shared constants and helpers for IDS Trusted Explainability Framework
# CIS 6372 – single source of truth for dataset config, loading, and trust score

import os
import pandas as pd

# -----------------------------
# Dataset configuration
# -----------------------------

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# -----------------------------
# Data loading
# -----------------------------

def load_nsl_kdd(file_path):
    """Load NSL-KDD CSV with correct column names. Raises FileNotFoundError if path missing."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, names=FEATURE_NAMES)


def convert_to_binary_labels(df):
    """Convert label column to binary: normal=0, attack=1. Returns a copy."""
    df_copy = df.copy()
    df_copy["label"] = df_copy["label"].apply(lambda x: 0 if x == "normal" else 1)
    return df_copy


def encode_features(df, feature_names=None):
    """
    Single source of truth for feature encoding: drop label/difficulty, one-hot encode,
    optionally align to model feature set. Use feature_names when aligning to saved model.
    """
    X = df.drop(columns=["label", "difficulty"])
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES)
    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)
    return X


def load_data(path, feature_names=None):
    """
    Load NSL-KDD data, encode with encode_features(), return (X, y).
    Use feature_names to align to model's feature set (required for inference/SHAP/LIME).
    """
    df = load_nsl_kdd(path)
    df = convert_to_binary_labels(df)
    X = encode_features(df, feature_names=feature_names)
    y = df["label"]
    return X, y

# -----------------------------
# Trust score (SHAP–LIME agreement)
# -----------------------------

def clean_lime_feature_name(lime_str):
    """
    LIME returns strings like 'src_bytes <= 0.00' or 'service_http > 0.5'.
    Extract the base feature name so we compare the same entities as SHAP.
    """
    return lime_str.split()[0] if isinstance(lime_str, str) and lime_str.strip() else lime_str


def compute_trust_score(shap_feature_names, lime_feature_names, top_k=5):
    """
    Trust Score = |Top-5 overlap| / 5 × 100 (unified definition across repo).
    Returns (trust_score_pct, overlap_set, trust_level, flag_low).
    Cutoffs: HIGH >= 4 overlap, MEDIUM >= 2, LOW otherwise.
    """
    lime_cleaned = [clean_lime_feature_name(s) for s in lime_feature_names[:top_k]]
    shap_set = set(shap_feature_names[:top_k])
    lime_set = set(lime_cleaned)
    overlap = shap_set & lime_set
    overlap_count = len(overlap)
    trust_score_pct = (overlap_count / top_k) * 100.0
    if overlap_count >= 4:
        trust_level = "HIGH"
    elif overlap_count >= 2:
        trust_level = "MEDIUM"
    else:
        trust_level = "LOW"
    flag_low = trust_level == "LOW"
    return trust_score_pct, overlap, trust_level, flag_low
