import pandas as pd

from backend.app.data.preparation import load_and_prepare_dataset
from backend.app.nlp.aspect_extraction import extract_aspects


def test_real_reviews() -> None:
    """
    Test aspect extraction on real reviews from
    the project's 21,000-review Amazon dataset.
    """

    print("Loading the real Amazon review dataset...")

    dataset = load_and_prepare_dataset()

    print(f"Dataset loaded successfully.")
    print(f"Total reviews: {len(dataset)}")

    print("\nSelecting sample reviews...")

    # Select 10 reviews from different positions in
    # the dataset instead of taking only the first 10.
    sample_indexes = [
        0,
        100,
        500,
        1000,
        2500,
        5000,
        7500,
        10000,
        15000,
        20000,
    ]

    sample_indexes = [
        index
        for index in sample_indexes
        if index < len(dataset)
    ]

    samples = dataset.iloc[sample_indexes]

    print(f"Testing {len(samples)} real reviews.")

    print("\n" + "=" * 80)

    for number, (_, row) in enumerate(
        samples.iterrows(),
        start=1,
    ):

        review = row["REVIEW_TEXT"]

        aspects = extract_aspects(review)

        print(f"\nREVIEW {number}")
        print("-" * 80)

        print("Original review:")
        print(review)

        print("\nExtracted aspects:")

        if aspects:
            for aspect in aspects:
                print(f"  - {aspect}")
        else:
            print("  No aspects detected.")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_real_reviews()