from pathlib import Path

import pandas as pd

from backend.app.data.preparation import load_and_prepare_dataset
from backend.app.nlp.preprocessing import preprocess_review


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "amazon_reviews_processed.csv"


def process_dataset() -> None:
    """Preprocess all reviews and save the processed dataset."""

    print("Loading original dataset...")

    dataset = load_and_prepare_dataset()

    print(f"Loaded {len(dataset)} reviews.")

    print("\nPreprocessing review text...")

    processed_results = dataset["REVIEW_TEXT"].apply(preprocess_review)

    dataset["CLEANED_TEXT"] = processed_results.apply(
        lambda result: result["cleaned_text"]
    )

    dataset["PROCESSED_TEXT"] = processed_results.apply(
        lambda result: result["processed_text"]
    )

    # Make sure the output directory exists.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the processed dataset.
    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nProcessing completed successfully.")
    print(f"Total reviews processed: {len(dataset)}")
    print(f"Output file: {OUTPUT_PATH}")

    print("\nOutput columns:")
    print(dataset.columns.tolist())


if __name__ == "__main__":
    process_dataset()