from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from backend.app.ml.feature_engineering import FeatureEngineer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon_reviews_processed.csv"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIRECTORY / "svm_model.joblib"
FEATURE_ENGINEER_PATH = MODEL_DIRECTORY / "feature_engineer.joblib"


def load_dataset() -> pd.DataFrame:
    """Load the processed Amazon review dataset."""

    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: "
            f"{PROCESSED_DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        PROCESSED_DATASET_PATH
    )

    required_columns = [
        "REVIEW_TEXT",
        "TARGET",
        "RATING",
        "VERIFIED_PURCHASE",
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

    return dataframe


def prepare_additional_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert rating and verified-purchase information
    into numerical features.
    """

    additional_features = pd.DataFrame(
        index=dataframe.index
    )

    additional_features["rating"] = (
        dataframe["RATING"]
        .astype(float)
    )

    additional_features["verified_purchase"] = (
        dataframe["VERIFIED_PURCHASE"]
        .map(
            {
                "Y": 1.0,
                "N": 0.0,
            }
        )
    )

    if additional_features[
        "verified_purchase"
    ].isnull().any():

        raise ValueError(
            "Unexpected VERIFIED_PURCHASE value found."
        )

    return additional_features


def combine_features(
    absa_text_features,
    additional_features: pd.DataFrame,
):
    """
    Combine TF-IDF/ABSA features with
    rating and verified-purchase features.
    """

    from scipy.sparse import csr_matrix, hstack

    additional_matrix = csr_matrix(
        additional_features.to_numpy(
            dtype=float
        )
    )

    return hstack(
        [
            absa_text_features,
            additional_matrix,
        ],
        format="csr",
    )


def split_dataset(
    dataframe: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split the dataset into training and testing sets."""

    X = dataframe.drop(
        columns=["TARGET"]
    )

    y = dataframe["TARGET"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def train_model() -> None:
    """Train and save the SVM fake-review classifier."""

    print("=" * 70)
    print("SVM FAKE REVIEW CLASSIFIER TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------

    dataframe = load_dataset()

    print(
        f"\nTotal reviews: {len(dataframe)}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        dataframe["TARGET"].value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------------
    # 2. Split before fitting TF-IDF
    # ---------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(dataframe)

    print(
        f"\nTraining reviews: {len(X_train)}"
    )

    print(
        f"Testing reviews: {len(X_test)}"
    )

    # ---------------------------------------------------------------
    # 3. Create feature engineer
    # ---------------------------------------------------------------

    feature_engineer = FeatureEngineer()

    # ---------------------------------------------------------------
    # 4. Fit TF-IDF ONLY on training data
    # ---------------------------------------------------------------

    print(
        "\nBuilding training features..."
    )

    train_text_absa_features, train_absa = (
        feature_engineer.fit_transform(
            X_train
        )
    )

    # ---------------------------------------------------------------
    # 5. Transform test data using the
    #    already-fitted TF-IDF vectorizer
    # ---------------------------------------------------------------

    print(
        "Building testing features..."
    )

    test_text_absa_features, test_absa = (
        feature_engineer.transform(
            X_test
        )
    )

    # ---------------------------------------------------------------
    # 6. Prepare rating and verified-purchase
    # ---------------------------------------------------------------

    train_additional_features = (
        prepare_additional_features(
            X_train
        )
    )

    test_additional_features = (
        prepare_additional_features(
            X_test
        )
    )

    # ---------------------------------------------------------------
    # 7. Combine all feature groups
    # ---------------------------------------------------------------

    X_train_features = combine_features(
        train_text_absa_features,
        train_additional_features,
    )

    X_test_features = combine_features(
        test_text_absa_features,
        test_additional_features,
    )

    print(
        "\nTraining feature matrix:"
    )

    print(
        X_train_features.shape
    )

    print(
        "Testing feature matrix:"
    )

    print(
        X_test_features.shape
    )

    # ---------------------------------------------------------------
    # 8. Train SVM
    # ---------------------------------------------------------------

    print(
        "\nTraining Support Vector Machine..."
    )

    classifier = SVC(
        kernel="linear",
        probability=True,
        random_state=42,
    )

    classifier.fit(
        X_train_features,
        y_train,
    )

    print(
        "SVM training completed."
    )

    # ---------------------------------------------------------------
    # 9. Create model directory
    # ---------------------------------------------------------------

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # 10. Save trained classifier
    # ---------------------------------------------------------------

    joblib.dump(
        classifier,
        MODEL_PATH,
    )

    # ---------------------------------------------------------------
    # 11. Save feature engineer
    # ---------------------------------------------------------------

    joblib.dump(
        feature_engineer,
        FEATURE_ENGINEER_PATH,
    )

    print(
        f"\nSVM model saved to:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nFeature engineer saved to:"
        f"\n{FEATURE_ENGINEER_PATH}"
    )

    # ---------------------------------------------------------------
    # 12. Basic prediction check
    # ---------------------------------------------------------------

    predictions = classifier.predict(
        X_test_features[:10]
    )

    print(
        "\nFirst 10 test predictions:"
    )

    print(predictions)

    print(
        "\nTraining pipeline completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    train_model()