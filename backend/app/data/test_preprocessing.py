from backend.app.data.preparation import load_and_prepare_dataset
from backend.app.nlp.preprocessing import preprocess_review


def main():
    # Load the real Amazon review dataset.
    dataset = load_and_prepare_dataset()

    print("Dataset loaded successfully.")
    print(f"Total reviews: {len(dataset)}")

    print("\n" + "=" * 80)
    print("TESTING PREPROCESSING ON REAL REVIEWS")
    print("=" * 80)

    # Test the first three reviews.
    for index, review in dataset.head(3).iterrows():
        result = preprocess_review(review["REVIEW_TEXT"])

        print(f"\nReview {index + 1}")
        print("-" * 80)

        print("Original:")
        print(review["REVIEW_TEXT"])

        print("\nCleaned:")
        print(result["cleaned_text"])

        print("\nProcessed:")
        print(result["processed_text"])

        print("\nLabel:")
        print(review["LABEL"])

        print("\nTarget:")
        print(review["TARGET"])


if __name__ == "__main__":
    main()