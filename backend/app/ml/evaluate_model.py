from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from backend.app.ml.train_model import (
    combine_features,
    load_dataset,
    prepare_additional_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "svm_model.joblib"
FEATURE_ENGINEER_PATH = (
    PROJECT_ROOT / "models" / "feature_engineer.joblib"
)


def evaluate_model() -> None:
    """Evaluate the trained SVM classifier on the held-out test set."""

    print("=" * 70)
    print("SVM FAKE REVIEW CLASSIFIER EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Check saved model files
    # ---------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SVM model not found at: {MODEL_PATH}"
        )

    if not FEATURE_ENGINEER_PATH.exists():
        raise FileNotFoundError(
            f"Feature engineer not found at: "
            f"{FEATURE_ENGINEER_PATH}"
        )

    # ---------------------------------------------------------------
    # 2. Load dataset and saved components
    # ---------------------------------------------------------------

    dataframe = load_dataset()

    classifier = joblib.load(MODEL_PATH)

    feature_engineer = joblib.load(
        FEATURE_ENGINEER_PATH
    )

    print(
        f"\nTotal reviews: {len(dataframe)}"
    )

    # ---------------------------------------------------------------
    # 3. Recreate the same train/test split
    # ---------------------------------------------------------------

    X = dataframe.drop(
        columns=["TARGET"]
    )

    y = dataframe["TARGET"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"Training reviews: {len(X_train)}"
    )

    print(
        f"Testing reviews: {len(X_test)}"
    )

    # ---------------------------------------------------------------
    # 4. Transform the test reviews
    # ---------------------------------------------------------------

    print(
        "\nBuilding test features..."
    )

    test_text_absa_features, _ = (
        feature_engineer.transform(
            X_test
        )
    )

    test_additional_features = (
        prepare_additional_features(
            X_test
        )
    )

    X_test_features = combine_features(
        test_text_absa_features,
        test_additional_features,
    )

    print(
        f"Test feature matrix: "
        f"{X_test_features.shape}"
    )

    # ---------------------------------------------------------------
    # 5. Generate predictions
    # ---------------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    predictions = classifier.predict(
        X_test_features
    )

    # ---------------------------------------------------------------
    # 6. Calculate evaluation metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # 7. Display metrics
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"\nAccuracy : {accuracy:.4f}"
        f" ({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f}"
        f" ({precision * 100:.2f}%)"
    )

    print(
        f"Recall   : {recall:.4f}"
        f" ({recall * 100:.2f}%)"
    )

    print(
        f"F1-score : {f1:.4f}"
        f" ({f1 * 100:.2f}%)"
    )

    # ---------------------------------------------------------------
    # 8. Classification report
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Genuine",
                "Fake / Deceptive",
            ],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------------
    # 9. Confusion matrix
    # ---------------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\nRows = Actual class"
        "\nColumns = Predicted class"
    )

    print(
        "\n                 Genuine  Fake"
    )

    print(
        f"Genuine          {matrix[0, 0]:7d}"
        f"  {matrix[0, 1]:7d}"
    )

    print(
        f"Fake / Deceptive {matrix[1, 0]:7d}"
        f"  {matrix[1, 1]:7d}"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_model()