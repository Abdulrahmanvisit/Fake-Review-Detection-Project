from pathlib import Path

import pandas as pd


# Project root:
# fake-review-detection/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Original dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "original" / "amazon_reviews.txt"

# Columns required by the project
REQUIRED_COLUMNS = [
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


def load_dataset() -> pd.DataFrame:
    """Load and validate the original Amazon review dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    # The original dataset is tab-separated.
    dataframe = pd.read_csv(
        DATASET_PATH,
        sep="\t",
    )

    # Check that all expected columns exist.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return dataframe


if __name__ == "__main__":
    dataset = load_dataset()

    print("Dataset loaded successfully.")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Number of reviews: {len(dataset)}")
    print(f"Number of columns: {len(dataset.columns)}")
    print(f"Columns: {dataset.columns.tolist()}")
    print("\nLabel distribution:")
    print(dataset["LABEL"].value_counts())