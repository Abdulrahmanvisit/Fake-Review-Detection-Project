import pandas as pd

from backend.app.nlp.aspect_extraction import extract_aspects
from backend.app.nlp.sentiment import classify_review_aspects


DATASET_PATH = "data/original/amazon_reviews.txt"


def run_real_review_test() -> None:
    """Test aspect-level sentiment on real Amazon reviews."""

    print("=" * 80)
    print("REAL AMAZON REVIEW SENTIMENT TEST")
    print("=" * 80)

    dataset = pd.read_csv(
        DATASET_PATH,
        sep="\t",
    )

    sample = dataset.sample(
        n=10,
        random_state=42,
    )

    for index, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):
        review = str(row["REVIEW_TEXT"])

        aspects = extract_aspects(review)

        sentiment_results = classify_review_aspects(
            review_text=review,
            aspects=aspects,
        )

        print(f"\n{'-' * 80}")
        print(f"REVIEW {index}")
        print(f"{'-' * 80}")

        print(f"Review:")
        print(review)

        print("\nExtracted aspects:")

        if not aspects:
            print("  No aspects detected.")
            continue

        for result in sentiment_results:
            print(
                f"  - {result['aspect']}: "
                f"{result['sentiment']} "
                f"(score={result['score']})"
            )

    print("\n" + "=" * 80)
    print("REAL REVIEW TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_real_review_test()