import re
from typing import List


# ---------------------------------------------------------------------------
# Controlled Aspect Vocabulary
# ---------------------------------------------------------------------------
#
# The extractor deliberately uses a controlled vocabulary rather than
# generating arbitrary neighbouring words. This reduces false-positive
# aspects while still allowing common product/review aspects to be detected.
# ---------------------------------------------------------------------------

KNOWN_ASPECTS = {
    # General product aspects
    "quality",
    "price",
    "cost",
    "size",
    "colour",
    "color",
    "design",
    "delivery",
    "shipping",
    "packaging",
    "service",
    "support",
    "performance",
    "speed",
    "material",
    "texture",
    "taste",
    "smell",
    "scent",
    "comfort",
    "fit",
    "style",
    "appearance",
    "value",
    "durability",
    "strength",
    "weight",
    "capacity",
    "quantity",
    "availability",
    "condition",

    # Electronics
    "battery",
    "batteries",
    "screen",
    "display",
    "camera",
    "lens",
    "sound",
    "audio",
    "volume",
    "speaker",
    "speakers",
    "keyboard",
    "mouse",
    "cursor",
    "computer",
    "laptop",
    "phone",
    "charger",
    "charging",
    "charge",
    "buttons",
    "button",
    "strap",
    "straps",
    "handle",
    "handles",
    "cover",
    "case",
    "box",
    "package",

    # Product features
    "alarm",
    "alarms",
    "setting",
    "settings",
    "function",
    "functions",
    "feature",
    "features",
    "threading",
    "mount",
    "mounts",

    # Clothing
    "shirt",
    "shirts",
    "polo",
    "dress",
    "clothing",

    # Household / physical products
    "cutter",
    "mat",
    "mats",
    "grater",
    "pillows",
    "pillow",
    "seat",
    "filling",
    "floor",
    "floors",
    "wall",
    "walls",
    "sticker",
    "stickers",
    "flower",
    "flowers",
    "butterfly",
    "butterflies",

    # Personal care / beauty
    "skin",
    "cellulite",
    "sensation",
    "reaction",
    "reactions",
    "effect",
    "tube",
    "ingredient",
    "ingredients",
    "circulation",

    # Other recognised review/product aspects
    "safety",
    "accuracy",
    "reliability",
    "life",
}


# ---------------------------------------------------------------------------
# High-Confidence Multi-Word Aspects
# ---------------------------------------------------------------------------

KNOWN_ASPECT_PHRASES = {
    "battery life",
    "sound quality",
    "camera quality",
    "build quality",
    "picture quality",
    "image quality",
    "video quality",
    "screen quality",
    "display quality",
    "customer service",
    "delivery time",
    "shipping time",
    "battery performance",
    "sound performance",
    "skin care",
    "pizza cutter",
    "tarot cards",
    "polo shirt",
    "product quality",
    "product price",
    "charging time",
    "battery charger",
}


# ---------------------------------------------------------------------------
# Contextual False-Positive Rules
# ---------------------------------------------------------------------------
#
# Some words can be aspects in one context and ordinary words in another.
#
# Example:
#     "The watch has comfortable straps."
#          -> watch is an aspect.
#
#     "I watched this movie."
#          -> watched/watch is a verb, not a product aspect.
#
# The rules below prevent common contextual false positives.
# ---------------------------------------------------------------------------

CONTEXT_EXCLUSIONS = {
    "watch": {
        "watched",
        "watching",
        "watch",
    },
}


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean review text before aspect extraction.

    HTML tags and unnecessary punctuation are removed while preserving
    normal alphabetic characters and numbers.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Decode common HTML entities represented in review text.
    text = text.replace(
        "&#34;",
        '"',
    )

    text = text.replace(
        "&quot;",
        '"',
    )

    text = text.replace(
        "&amp;",
        "and",
    )

    text = text.lower()

    # Keep letters, numbers, spaces and apostrophes.
    text = re.sub(
        r"[^a-z0-9'\s-]",
        " ",
        text,
    )

    # Normalise whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Tokenise cleaned review text into simple word tokens.
    """
    return re.findall(
        r"[a-z]+",
        text.lower(),
    )


# ---------------------------------------------------------------------------
# Multi-Word Aspect Extraction
# ---------------------------------------------------------------------------

def extract_known_phrases(
    tokens: List[str],
) -> List[str]:
    """
    Extract only recognised high-confidence multi-word aspects.
    """
    aspects = []

    token_text = " ".join(tokens)

    for phrase in sorted(
        KNOWN_ASPECT_PHRASES,
        key=lambda value: len(value.split()),
        reverse=True,
    ):
        if re.search(
            rf"\b{re.escape(phrase)}\b",
            token_text,
        ):
            aspects.append(phrase)

    return aspects


# ---------------------------------------------------------------------------
# Context-Aware Single-Word Aspect Extraction
# ---------------------------------------------------------------------------

def is_contextually_valid_aspect(
    tokens: List[str],
    index: int,
    aspect: str,
) -> bool:
    """
    Determine whether a single-word aspect is being used as an aspect
    rather than as an ordinary word or verb.
    """

    # ---------------------------------------------------------------
    # Special handling for "watch"
    # ---------------------------------------------------------------
    #
    # "watch" can represent a physical product:
    #
    #     "The watch has good straps."
    #
    # But it can also be a verb:
    #
    #     "I watch this movie."
    #
    # The surrounding words help distinguish these cases.
    # ---------------------------------------------------------------

    if aspect == "watch":

        previous_word = (
            tokens[index - 1]
            if index > 0
            else ""
        )

        next_word = (
            tokens[index + 1]
            if index + 1 < len(tokens)
            else ""
        )

        # Common verb constructions.
        if next_word in {
            "this",
            "that",
            "the",
            "a",
            "an",
            "movies",
            "movie",
            "videos",
            "video",
        }:
            return False

        if previous_word in {
            "i",
            "we",
            "you",
            "they",
            "he",
            "she",
            "to",
            "can",
            "will",
            "would",
            "like",
            "love",
        }:
            return False

    return True


def extract_single_word_aspects(
    tokens: List[str],
    protected_phrase_words: set,
) -> List[str]:
    """
    Extract recognised single-word aspects from the controlled vocabulary.
    """
    aspects = []

    for index, token in enumerate(tokens):

        if token not in KNOWN_ASPECTS:
            continue

        # Do not extract words that are already part of a recognised
        # multi-word aspect.
        if token in protected_phrase_words:
            continue

        if not is_contextually_valid_aspect(
            tokens=tokens,
            index=index,
            aspect=token,
        ):
            continue

        aspects.append(token)

    return aspects


# ---------------------------------------------------------------------------
# Duplicate and Redundancy Removal
# ---------------------------------------------------------------------------

def remove_redundant_aspects(
    aspects: List[str],
) -> List[str]:
    """
    Remove duplicate and unnecessary component aspects.

    Example:
        battery life
        battery

    becomes:

        battery life
    """
    unique_aspects = []

    for aspect in aspects:
        if aspect not in unique_aspects:
            unique_aspects.append(aspect)

    # Prefer multi-word aspects over their single-word components.
    final_aspects = []

    for aspect in unique_aspects:

        is_component = False

        for other_aspect in unique_aspects:

            if aspect == other_aspect:
                continue

            other_words = set(
                other_aspect.split()
            )

            if (
                " " not in aspect
                and aspect in other_words
            ):
                is_component = True
                break

        if not is_component:
            final_aspects.append(aspect)

    return final_aspects


# ---------------------------------------------------------------------------
# Main Aspect Extraction Function
# ---------------------------------------------------------------------------

def extract_aspects(
    text: str,
) -> List[str]:
    """
    Extract recognised aspects from a review.

    The process is:

        1. Clean text
        2. Tokenise text
        3. Extract known multi-word aspects
        4. Extract known single-word aspects
        5. Remove redundant aspects

    The approach is intentionally conservative to reduce false-positive
    aspect extraction.
    """
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    tokens = tokenize(cleaned_text)

    if not tokens:
        return []

    # ---------------------------------------------------------------
    # Extract multi-word aspects first.
    # ---------------------------------------------------------------

    phrase_aspects = extract_known_phrases(
        tokens
    )

    # Keep track of words already used by recognised phrases.
    protected_phrase_words = set()

    for phrase in phrase_aspects:
        protected_phrase_words.update(
            phrase.split()
        )

    # ---------------------------------------------------------------
    # Extract single-word aspects.
    # ---------------------------------------------------------------

    single_word_aspects = extract_single_word_aspects(
        tokens=tokens,
        protected_phrase_words=protected_phrase_words,
    )

    # ---------------------------------------------------------------
    # Combine results.
    # ---------------------------------------------------------------

    aspects = (
        phrase_aspects
        + single_word_aspects
    )

    # ---------------------------------------------------------------
    # Remove duplicates/redundancy.
    # ---------------------------------------------------------------

    return remove_redundant_aspects(
        aspects
    )


# ---------------------------------------------------------------------------
# Simple Manual Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    test_reviews = [
        (
            "The battery life is excellent and "
            "the screen is very clear."
        ),
        (
            "The sound quality is amazing but "
            "the price is high."
        ),
        (
            "Delivery was fast and the packaging was good."
        ),
        (
            "I purchased this polo shirt for my husband "
            "and it fits very well."
        ),
        (
            "This charger brought back my batteries "
            "and the display is excellent."
        ),
        (
            "I have watched this movie and it was amazing."
        ),
        (
            "This is the best hand grater I have ever owned."
        ),
    ]

    print("=" * 70)
    print("ASPECT EXTRACTION TEST")
    print("=" * 70)

    for index, review in enumerate(
        test_reviews,
        start=1,
    ):
        print(f"\nReview {index}:")
        print(review)

        aspects = extract_aspects(review)

        print("Aspects:")

        if not aspects:
            print("  No aspects detected.")
        else:
            for aspect in aspects:
                print(f"  - {aspect}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)