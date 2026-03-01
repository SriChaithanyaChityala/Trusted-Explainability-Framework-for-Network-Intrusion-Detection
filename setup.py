# Week 1: Environment Setup & NSL-KDD Data Loading
# CIS 6372 – Trusted Explainability Framework for IDS

from utils import load_nsl_kdd, convert_to_binary_labels

# -----------------------------
# Main execution
# -----------------------------

if __name__ == "__main__":

    # Update paths if needed
    TRAIN_PATH = "data/raw/KDDTrain+.txt"
    TEST_PATH = "data/raw/KDDTest+.txt"

    print("Loading NSL-KDD dataset...")

    train_df = load_nsl_kdd(TRAIN_PATH)
    test_df = load_nsl_kdd(TEST_PATH)

    train_df = convert_to_binary_labels(train_df)
    test_df = convert_to_binary_labels(test_df)

    print("\nDataset loaded successfully")
    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    print("\nLabel distribution (Training):")
    print(train_df["label"].value_counts())

    print("\nSample rows:")
    print(train_df.head())
