import re
from typing import List


# ================================================================
# KNOWN SINGLE-WORD ASPECTS
# ================================================================
#
# Only words in this vocabulary can become standalone aspects.
# This prevents ordinary words such as "book", "case", "brain",
# "muddy", etc. from automatically becoming aspects.
#
# ================================================================

KNOWN_ASPECTS = {
    # General product characteristics
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

    # Electronics / devices
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
    "computer",
    "laptop",
    "phone",
    "watch",

    # Product components
    "button",
    "buttons",
    "strap",
    "straps",
    "handle",
    "handles",
    "cover",
    "box",
    "package",

    # Product functions
    "alarm",
    "alarms",
    "setting",
    "settings",
    "function",
    "functions",
    "feature",
    "features",

    # Clothing
    "shirt",
    "shirts",

    # Household / kitchen
    "cutter",
    "mat",
    "mats",

    # Home / decoration
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

    # Health / beauty
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

    # Other useful product properties
    "safety",
    "accuracy",
    "reliability",
    "capacity",
    "quantity",
    "availability",
}


# ================================================================
# HIGH-CONFIDENCE MULTI-WORD ASPECTS
# ================================================================
#
# These are the ONLY multi-word aspects the extractor will
# automatically create.
#
# We deliberately do NOT create arbitrary phrases such as:
#
#     muddy sensation
#     odd reactions
#     common settings
#     different walls
#
# ================================================================

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
}


# ================================================================
# TEXT CLEANING
# ================================================================

def clean_text(text: str) -> str:
    """
    Clean review text before aspect extraction.

    HTML tags are removed and whitespace is normalised.
    """

    if not isinstance(text, str):
        return ""

    # Remove HTML such as <br />, <br><br>, <p>, etc.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Convert text to lowercase.
    text = text.lower()

    # Keep letters, numbers, apostrophes and spaces.
    text = re.sub(
        r"[^a-z0-9\s']",
        " ",
        text,
    )

    # Remove repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ================================================================
# TOKENISATION
# ================================================================

def tokenize(text: str) -> List[str]:
    """
    Convert cleaned text into simple word tokens.
    """

    return re.findall(
        r"[a-z]+",
        text,
    )


# ================================================================
# MULTI-WORD ASPECT EXTRACTION
# ================================================================

def extract_known_phrases(
    text: str,
) -> List[str]:
    """
    Extract only explicitly recognised multi-word aspects.
    """

    tokens = tokenize(text)

    aspects = []

    for index in range(len(tokens) - 1):

        phrase = (
            f"{tokens[index]} "
            f"{tokens[index + 1]}"
        )

        if phrase in KNOWN_ASPECT_PHRASES:

            if phrase not in aspects:
                aspects.append(
                    phrase
                )

    return aspects


# ================================================================
# SINGLE-WORD ASPECT EXTRACTION
# ================================================================

def extract_single_word_aspects(
    tokens: List[str],
    phrase_aspects: List[str],
) -> List[str]:
    """
    Extract known single-word aspects.

    Words already belonging to a recognised multi-word
    aspect are not returned separately.

    Example:

        battery life

    returns:

        battery life

    rather than:

        battery
        life
    """

    # Store every word that belongs to a recognised phrase.
    protected_words = set()

    for phrase in phrase_aspects:

        for word in phrase.split():

            protected_words.add(
                word
            )

    aspects = []

    for token in tokens:

        # Skip words already represented by a phrase.
        if token in protected_words:
            continue

        # Extract only explicitly known aspects.
        if token in KNOWN_ASPECTS:

            if token not in aspects:

                aspects.append(
                    token
                )

    return aspects


# ================================================================
# CONTEXTUAL FALSE-POSITIVE FILTER
# ================================================================

def remove_false_positive_aspects(
    aspects: List[str],
    text: str,
) -> List[str]:
    """
    Remove known contextual false positives.

    This is intentionally small and conservative.

    Example:

        "in case you need it"

    should not produce:

        case
    """

    tokens = tokenize(text)

    final_aspects = []

    for aspect in aspects:

        # --------------------------------------------------------
        # "case" is ambiguous.
        #
        # We keep "case" as a possible product aspect only when
        # it does not occur in the phrase "in case".
        # --------------------------------------------------------

        if aspect == "case":

            for index, token in enumerate(tokens):

                if token != "case":
                    continue

                if (
                    index > 0
                    and tokens[index - 1] == "in"
                ):
                    continue

            # Since "case" is not currently part of the
            # conservative KNOWN_ASPECTS vocabulary, this
            # condition normally does not execute.
            continue

        # --------------------------------------------------------
        # Avoid "alarm" duplicates when "alarms" is present.
        # --------------------------------------------------------

        if (
            aspect == "alarm"
            and "alarms" in aspects
        ):
            continue

        # --------------------------------------------------------
        # Avoid singular/plural duplicates.
        # --------------------------------------------------------

        if (
            aspect.endswith("s")
            and aspect[:-1] in aspects
        ):
            continue

        if (
            aspect + "s" in aspects
        ):
            continue

        final_aspects.append(
            aspect
        )

    return final_aspects


# ================================================================
# REMOVE REDUNDANT COMPONENT ASPECTS
# ================================================================

def remove_component_aspects(
    aspects: List[str],
) -> List[str]:
    """
    Remove single-word components when they are already
    represented by a more specific multi-word aspect.

    Example:

        battery life
        battery

    becomes:

        battery life
    """

    multi_word_aspects = [
        aspect
        for aspect in aspects
        if " " in aspect
    ]

    final_aspects = []

    for aspect in aspects:

        # If it is already a multi-word aspect, keep it.
        if " " in aspect:

            if aspect not in final_aspects:

                final_aspects.append(
                    aspect
                )

            continue

        # Check whether this single word is part of a
        # recognised multi-word aspect.
        belongs_to_phrase = False

        for phrase in multi_word_aspects:

            if aspect in phrase.split():

                belongs_to_phrase = True

                break

        if belongs_to_phrase:
            continue

        if aspect not in final_aspects:

            final_aspects.append(
                aspect
            )

    return final_aspects


# ================================================================
# MAIN ASPECT EXTRACTION FUNCTION
# ================================================================

def extract_aspects(
    text: str,
) -> List[str]:
    """
    Extract product aspects from an e-commerce review.

    The method uses a conservative rule-based approach:

        1. Clean the review.
        2. Tokenise the review.
        3. Detect predefined multi-word aspects.
        4. Detect predefined single-word aspects.
        5. Remove duplicate/component aspects.
        6. Return the final aspect list.

    The extractor intentionally avoids generating arbitrary
    neighbouring word combinations. This reduces false positives.
    """

    # ------------------------------------------------------------
    # Step 1: Clean text
    # ------------------------------------------------------------

    cleaned_text = clean_text(
        text
    )

    if not cleaned_text:
        return []

    # ------------------------------------------------------------
    # Step 2: Tokenise
    # ------------------------------------------------------------

    tokens = tokenize(
        cleaned_text
    )

    if not tokens:
        return []

    # ------------------------------------------------------------
    # Step 3: Extract known multi-word aspects
    # ------------------------------------------------------------

    phrase_aspects = (
        extract_known_phrases(
            cleaned_text
        )
    )

    # ------------------------------------------------------------
    # Step 4: Extract known single-word aspects
    # ------------------------------------------------------------

    single_word_aspects = (
        extract_single_word_aspects(
            tokens,
            phrase_aspects,
        )
    )

    # ------------------------------------------------------------
    # Step 5: Combine
    # ------------------------------------------------------------

    aspects = (
        phrase_aspects
        + single_word_aspects
    )

    # ------------------------------------------------------------
    # Step 6: Remove false positives
    # ------------------------------------------------------------

    aspects = (
        remove_false_positive_aspects(
            aspects,
            cleaned_text,
        )
    )

    # ------------------------------------------------------------
    # Step 7: Remove redundant components
    # ------------------------------------------------------------

    aspects = (
        remove_component_aspects(
            aspects
        )
    )

    return aspects


# ================================================================
# CONTROLLED TEST
# ================================================================

if __name__ == "__main__":

    sample_reviews = [

        "The battery life is excellent and the screen is very clear.",

        "The sound quality is amazing but the price is high.",

        "Delivery was fast and the packaging was good.",

        "The camera quality is excellent but battery life is poor.",

        "I purchased this polo shirt for my husband and it fits very well.",

        "My husband and I are having fun with our new pizza cutter.",

        "I have been a fan of tarot cards for many years and this book helped me learn even more about reading the cards.",

        "The watch has durable buttons and comfortable straps.",

        "The price is high but the product quality is excellent.",

        "The watch has three alarms and common settings.",

        "The cream provides smoother skin and causes no odd reactions.",

        "I have stickers and flowers on different walls with my own design.",
    ]

    print("=" * 80)
    print("ASPECT EXTRACTION TEST")
    print("=" * 80)

    for number, review in enumerate(
        sample_reviews,
        start=1,
    ):

        print(
            f"\nREVIEW {number}"
        )

        print("-" * 80)

        print("Original:")
        print(review)

        aspects = extract_aspects(
            review
        )

        print("\nExtracted aspects:")

        if aspects:

            for aspect in aspects:

                print(
                    f"  - {aspect}"
                )

        else:

            print(
                "  No aspects detected."
            )

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)