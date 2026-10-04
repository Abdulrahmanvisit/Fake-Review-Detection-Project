from pathlib import Path

import pandas as pd


# Project root:
# fake-review-detection/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Original dataset
DATASET_PATH = PROJECT_ROOT / "data" / "original" / "amazon_reviews.txt"


# Dataset label mapping
LABEL_MAPPING = {
    "__label1__": 1,  # Fake / deceptive
    "__label2__": 0,  # Genuine
}


def load_and_prepare_dataset() -> pd.DataFrame:
    """
    Load the original Amazon review dataset and prepare
    the label column for machine learning.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    # Load the original tab-separated dataset.
    dataframe = pd.read_csv(
        DATASET_PATH,
        sep="\t",
    )

    # Check required columns.
    required_columns = [
        "DOC_ID",
        "LABEL",
        "RATING",
        "VERIFIED_PURCHASE",
        "PRODUCT_CATEGORY",
        "PRODUCT_ID",
        "PRODUCT_TITLE",
        "REVIEW_TITLE",
        "REVIEW_TEXT",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check that all labels are known.
    unknown_labels = set(dataframe["LABEL"].unique()) - set(LABEL_MAPPING)

    if unknown_labels:
        raise ValueError(
            f"Unknown dataset labels found: {unknown_labels}"
        )

    # Create numerical target variable for machine learning.
    dataframe["TARGET"] = dataframe["LABEL"].map(LABEL_MAPPING)

    # Verify that the conversion did not create missing values.
    if dataframe["TARGET"].isnull().any():
        raise ValueError(
            "Some labels could not be converted to TARGET values."
        )

    return dataframe


if __name__ == "__main__":
    dataset = load_and_prepare_dataset()

    print("Dataset preparation successful.")
    print(f"Total reviews: {len(dataset)}")

    print("\nOriginal label distribution:")
    print(dataset["LABEL"].value_counts())

    print("\nTarget distribution:")
    print(dataset["TARGET"].value_counts().sort_index())

    print("\nTarget meaning:")
    print("0 = Genuine")
    print("1 = Fake / Deceptive")

    print("\nPrepared columns:")
    print(dataset.columns.tolist())