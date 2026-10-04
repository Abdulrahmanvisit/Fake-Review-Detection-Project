from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.nlp.aspect_extraction import extract_aspects
from backend.app.nlp.sentiment import classify_review_aspects


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon_reviews_processed.csv"
)


class FeatureEngineer:
    """
    Convert processed review text and ABSA information
    into numerical features for the SVM classifier.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            sublinear_tf=True,
        )

        self.is_fitted = False

    @staticmethod
    def safe_text(value: object) -> str:
        """Convert a dataset value into safe text."""

        if pd.isna(value):
            return ""

        return str(value)

    @staticmethod
    def aspect_features(text: str) -> Dict[str, float]:
        """
        Extract aspects and convert their aspect-level
        sentiment information into numerical features.
        """

        if not text:
            return {
                "aspect_count": 0.0,
                "positive_aspect_count": 0.0,
                "negative_aspect_count": 0.0,
                "neutral_aspect_count": 0.0,
                "sentiment_score": 0.0,
            }

        aspects = extract_aspects(text)

        if not aspects:
            return {
                "aspect_count": 0.0,
                "positive_aspect_count": 0.0,
                "negative_aspect_count": 0.0,
                "neutral_aspect_count": 0.0,
                "sentiment_score": 0.0,
            }

        sentiment_results = classify_review_aspects(
            review_text=text,
            aspects=aspects,
        )

        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_score = 0.0

        for result in sentiment_results:

            sentiment = result["sentiment"]
            score = float(result["score"])

            total_score += score

            if sentiment == "Positive":
                positive_count += 1

            elif sentiment == "Negative":
                negative_count += 1

            else:
                neutral_count += 1

        return {
            "aspect_count": float(len(aspects)),
            "positive_aspect_count": float(
                positive_count
            ),
            "negative_aspect_count": float(
                negative_count
            ),
            "neutral_aspect_count": float(
                neutral_count
            ),
            "sentiment_score": total_score,
        }

    def build_absa_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build numerical features from aspect-based
        sentiment analysis.
        """

        if "REVIEW_TEXT" not in dataframe.columns:
            raise ValueError(
                "The dataframe must contain REVIEW_TEXT."
            )

        feature_rows: List[Dict[str, float]] = []

        for _, row in dataframe.iterrows():

            review_text = self.safe_text(
                row["REVIEW_TEXT"]
            )

            features = self.aspect_features(
                review_text
            )

            feature_rows.append(features)

        return pd.DataFrame(
            feature_rows,
            index=dataframe.index,
        )

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Fit TF-IDF on training data and combine it
        with ABSA numerical features.
        """

        text_column = self.get_text_column(
            dataframe
        )

        texts = (
            dataframe[text_column]
            .fillna("")
            .astype(str)
        )

        # Convert review text into TF-IDF features.
        tfidf_features = (
            self.vectorizer.fit_transform(texts)
        )

        # Generate ABSA-derived numerical features.
        absa_features = self.build_absa_features(
            dataframe
        )

        absa_matrix = csr_matrix(
            absa_features.to_numpy(
                dtype=float
            )
        )

        # Combine textual and ABSA features.
        combined_features = hstack(
            [
                tfidf_features,
                absa_matrix,
            ],
            format="csr",
        )

        self.is_fitted = True

        return combined_features, absa_features

    def transform(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Transform new data using the already-fitted
        TF-IDF representation.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "FeatureEngineer must be fitted "
                "before transform() is called."
            )

        text_column = self.get_text_column(
            dataframe
        )

        texts = (
            dataframe[text_column]
            .fillna("")
            .astype(str)
        )

        # Use the vocabulary learned from training data.
        tfidf_features = (
            self.vectorizer.transform(texts)
        )

        # Generate ABSA features.
        absa_features = self.build_absa_features(
            dataframe
        )

        absa_matrix = csr_matrix(
            absa_features.to_numpy(
                dtype=float
            )
        )

        combined_features = hstack(
            [
                tfidf_features,
                absa_matrix,
            ],
            format="csr",
        )

        return combined_features, absa_features

    @staticmethod
    def get_text_column(
        dataframe: pd.DataFrame,
    ) -> str:
        """
        Select the processed review text column.
        """

        if "PROCESSED_TEXT" in dataframe.columns:
            return "PROCESSED_TEXT"

        if "CLEANED_TEXT" in dataframe.columns:
            return "CLEANED_TEXT"

        if "REVIEW_TEXT" in dataframe.columns:
            return "REVIEW_TEXT"

        raise ValueError(
            "No suitable review text column was found."
        )


def load_processed_dataset() -> pd.DataFrame:
    """Load the processed Amazon review dataset."""

    if not PROCESSED_DATASET_PATH.exists():

        raise FileNotFoundError(
            "Processed dataset was not found at: "
            f"{PROCESSED_DATASET_PATH}"
        )

    return pd.read_csv(
        PROCESSED_DATASET_PATH
    )


if __name__ == "__main__":

    print("=" * 70)
    print("FEATURE ENGINEERING TEST")
    print("=" * 70)

    dataset = load_processed_dataset()

    print(
        f"Dataset shape: {dataset.shape}"
    )

    # Small sample used only to verify
    # that the feature-engineering pipeline works.
    sample = dataset.head(100).copy()

    engineer = FeatureEngineer()

    feature_matrix, absa_features = (
        engineer.fit_transform(sample)
    )

    print(
        f"Rows tested: {len(sample)}"
    )

    print(
        f"Combined feature matrix shape: "
        f"{feature_matrix.shape}"
    )

    print("\nABSA feature columns:")

    for column in absa_features.columns:
        print(f"  - {column}")

    print("\nFirst five ABSA feature rows:")

    print(
        absa_features.head()
    )

    print(
        "\nFeature matrix type: "
        f"{type(feature_matrix).__name__}"
    )

    print(
        "\nFeature engineering test completed successfully."
    )

    print("=" * 70)