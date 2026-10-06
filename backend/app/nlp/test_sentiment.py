from backend.app.nlp.sentiment import classify_review_aspects


TEST_CASES = [
    {
        "review": (
            "The delivery was fast, but the price was high "
            "and the packaging was not great."
        ),
        "aspects": [
            "delivery",
            "price",
            "packaging",
        ],
        "expected": [
            "Positive",
            "Negative",
            "Negative",
        ],
    },
    {
        "review": (
            "The product quality is excellent "
            "and the battery is amazing."
        ),
        "aspects": [
            "quality",
            "battery",
        ],
        "expected": [
            "Positive",
            "Positive",
        ],
    },
    {
        "review": (
            "The delivery was slow "
            "and the packaging was poor."
        ),
        "aspects": [
            "delivery",
            "packaging",
        ],
        "expected": [
            "Negative",
            "Negative",
        ],
    },
    {
        "review": (
            "The quality is not good "
            "and the service is terrible."
        ),
        "aspects": [
            "quality",
            "service",
        ],
        "expected": [
            "Negative",
            "Negative",
        ],
    },
    {
        "review": (
            "The quality is high "
            "but the price is high."
        ),
        "aspects": [
            "quality",
            "price",
        ],
        "expected": [
            "Positive",
            "Negative",
        ],
    },
    {
        "review": (
            "The product is not bad "
            "and the delivery was fast."
        ),
        "aspects": [
            "product",
            "delivery",
        ],
        "expected": [
            "Positive",
            "Positive",
        ],
    },
    {
        "review": "The battery never worked.",
        "aspects": ["battery"],
        "expected": ["Negative"],
    },
    {
        "review": "The battery isn't good.",
        "aspects": ["battery"],
        "expected": ["Negative"],
    },
    {
        "review": "The battery lasts a long time.",
        "aspects": ["battery"],
        "expected": ["Positive"],
    },
    {
        "review": (
            "The screen quality is excellent, "
            "but the battery is poor."
        ),
        "aspects": ["screen quality", "battery"],
        "expected": ["Positive", "Negative"],
    },
    {
        "review": "The screen is excellent. The battery is okay.",
        "aspects": ["screen", "battery"],
        "expected": ["Positive", "Neutral"],
    },
    {
        "review": (
            "The battery is excellent at first, "
            "but the battery is poor now."
        ),
        "aspects": ["battery"],
        "expected": ["Neutral"],
    },
]


def run_tests() -> None:
    """Run controlled aspect-level sentiment tests."""
    total = len(TEST_CASES)
    passed = 0

    print("=" * 70)
    print("ASPECT-LEVEL SENTIMENT TEST")
    print("=" * 70)

    for index, test_case in enumerate(TEST_CASES, start=1):
        review = test_case["review"]
        aspects = test_case["aspects"]
        expected = test_case["expected"]

        results = classify_review_aspects(
            review_text=review,
            aspects=aspects,
        )

        predicted = [
            result["sentiment"]
            for result in results
        ]

        test_passed = predicted == expected

        if test_passed:
            passed += 1

        print(f"\nTest {index}")
        print(f"Review: {review}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(
            f"Status: {'PASS' if test_passed else 'FAIL'}"
        )

        for result in results:
            print(
                f"  {result['aspect']}: "
                f"{result['sentiment']} "
                f"(score={result['score']})"
            )

    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 70)

    if passed != total:
        raise AssertionError(
            "One or more sentiment tests failed."
        )


if __name__ == "__main__":
    run_tests()