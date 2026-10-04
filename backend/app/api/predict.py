from pathlib import Path

import joblib
import pandas as pd

from backend.app.ml.train_model import (
    combine_features,
    prepare_additional_features,
)
from backend.app.nlp.aspect_extraction import extract_aspects
from backend.app.nlp.sentiment import classify_review_aspects


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "svm_model.joblib"
FEATURE_ENGINEER_PATH = (
    PROJECT_ROOT / "models" / "feature_engineer.joblib"
)


def load_prediction_components():
    """Load the trained SVM model and feature engineer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    if not FEATURE_ENGINEER_PATH.exists():
        raise FileNotFoundError(
            f"Feature engineer not found at: "
            f"{FEATURE_ENGINEER_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    feature_engineer = joblib.load(
        FEATURE_ENGINEER_PATH
    )

    return model, feature_engineer


def predict_review(
    review_text: str,
    rating: float = 0.0,
    verified_purchase: str = "N",
) -> dict:
    """
    Analyse one review and return its
    aspects, sentiments and classification.
    """

    if not isinstance(review_text, str):
        raise ValueError(
            "Review text must be a string."
        )

    review_text = review_text.strip()

    if not review_text:
        raise ValueError(
            "Review text cannot be empty."
        )

    if rating < 0 or rating > 5:
        raise ValueError(
            "Rating must be between 0 and 5."
        )

    verified_purchase = (
        verified_purchase.strip().upper()
    )

    if verified_purchase not in {"Y", "N"}:
        raise ValueError(
            "verified_purchase must be 'Y' or 'N'."
        )

    # ---------------------------------------------------------------
    # 1. Extract aspects
    # ---------------------------------------------------------------

    aspects = extract_aspects(
        review_text
    )

    # ---------------------------------------------------------------
    # 2. Determine sentiment for each aspect
    # ---------------------------------------------------------------

    sentiment_results = classify_review_aspects(
        review_text=review_text,
        aspects=aspects,
    )

    # ---------------------------------------------------------------
    # 3. Prepare review data
    # ---------------------------------------------------------------

    review_data = pd.DataFrame(
        [
            {
                "REVIEW_TEXT": review_text,
                "RATING": rating,
                "VERIFIED_PURCHASE": verified_purchase,
            }
        ]
    )

    # ---------------------------------------------------------------
    # 4. Load trained components
    # ---------------------------------------------------------------

    model, feature_engineer = (
        load_prediction_components()
    )

    # ---------------------------------------------------------------
    # 5. Build text and ABSA features
    # ---------------------------------------------------------------

    text_absa_features, _ = (
        feature_engineer.transform(
            review_data
        )
    )

    # ---------------------------------------------------------------
    # 6. Build additional features
    # ---------------------------------------------------------------

    additional_features = (
        prepare_additional_features(
            review_data
        )
    )

    # ---------------------------------------------------------------
    # 7. Combine features
    # ---------------------------------------------------------------

    final_features = combine_features(
        text_absa_features,
        additional_features,
    )

    # ---------------------------------------------------------------
    # 8. Predict class
    # ---------------------------------------------------------------

    prediction = int(
        model.predict(
            final_features
        )[0]
    )

    # ---------------------------------------------------------------
    # 9. Calculate prediction confidence
    # ---------------------------------------------------------------

    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            final_features
        )[0]

        confidence = float(
            max(probabilities) * 100
        )

    # ---------------------------------------------------------------
    # 10. Convert class to readable label
    # ---------------------------------------------------------------

    classification = (
        "Fake / Deceptive"
        if prediction == 1
        else "Genuine"
    )

    # ---------------------------------------------------------------
    # 11. Return structured result
    # ---------------------------------------------------------------

    return {
        "review": review_text,
        "classification": classification,
        "prediction": prediction,
        "confidence": (
            round(confidence, 2)
            if confidence is not None
            else None
        ),
        "aspects": sentiment_results,
    }


if __name__ == "__main__":

    print("=" * 70)
    print("FAKE REVIEW PREDICTION TEST")
    print("=" * 70)

    test_review = (
        "The delivery was fast and the product quality "
        "was excellent, but the price was too high."
    )

    result = predict_review(
        review_text=test_review,
        rating=4,
        verified_purchase="Y",
    )

    print("\nReview:")
    print(result["review"])

    print("\nClassification:")
    print(result["classification"])

    print(
        f"Confidence: "
        f"{result['confidence']}%"
    )

    print("\nAspect Sentiment:")

    if result["aspects"]:

        for item in result["aspects"]:

            print(
                f"  - {item['aspect']}: "
                f"{item['sentiment']} "
                f"(score={item['score']})"
            )

    else:

        print("  No aspects detected.")

    print("\n" + "=" * 70)
    print("PREDICTION TEST COMPLETED")
    print("=" * 70)