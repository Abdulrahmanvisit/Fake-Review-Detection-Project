import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Sentiment Lexicons
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "beautiful",
    "brilliant",
    "comfortable",
    "excellent",
    "fantastic",
    "fast",
    "good",
    "great",
    "helpful",
    "impressive",
    "love",
    "lovely",
    "perfect",
    "pleasant",
    "positive",
    "reliable",
    "satisfied",
    "smooth",
    "strong",
    "superb",
    "useful",
    "wonderful",
}

NEGATIVE_WORDS = {
    "awful",
    "bad",
    "broken",
    "cheap",
    "difficult",
    "disappointing",
    "disappointed",
    "expensive",
    "hard",
    "horrible",
    "poor",
    "problem",
    "problems",
    "slow",
    "terrible",
    "uncomfortable",
    "useless",
    "weak",
    "worst",
}


# ---------------------------------------------------------------------------
# Negation, Intensifier and Diminisher Lexicons
# ---------------------------------------------------------------------------

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "hardly",
    "barely",
    "scarcely",
}

INTENSIFIERS = {
    "very": 2.0,
    "really": 1.5,
    "extremely": 2.0,
    "absolutely": 2.0,
    "highly": 1.5,
    "so": 1.5,
    "too": 1.5,
}

DIMINISHERS = {
    "slightly": 0.5,
    "somewhat": 0.5,
    "fairly": 0.75,
    "quite": 0.75,
}


# ---------------------------------------------------------------------------
# Aspect-Specific Sentiment Rules
# ---------------------------------------------------------------------------
#
# Some words have different meanings depending on the aspect.
#
# Example:
#     high quality  -> Positive
#     high price    -> Negative
#
# These rules make the classifier aspect-aware instead of relying only
# on a general positive/negative word list.
# ---------------------------------------------------------------------------

ASPECT_CONTEXT_RULES = {
    "price": {
        "positive": {
            "reasonable",
            "affordable",
            "cheap",
            "low",
            "fair",
        },
        "negative": {
            "high",
            "expensive",
            "costly",
            "overpriced",
        },
    },
    "cost": {
        "positive": {
            "reasonable",
            "affordable",
            "cheap",
            "low",
            "fair",
        },
        "negative": {
            "high",
            "expensive",
            "costly",
            "overpriced",
        },
    },
    "quality": {
        "positive": {
            "high",
            "excellent",
            "great",
            "good",
            "superior",
        },
        "negative": {
            "poor",
            "low",
            "bad",
            "terrible",
        },
    },
    "performance": {
        "positive": {
            "high",
            "excellent",
            "great",
            "good",
            "strong",
        },
        "negative": {
            "poor",
            "low",
            "bad",
            "weak",
        },
    },
    "delivery": {
        "positive": {
            "fast",
            "quick",
            "prompt",
            "early",
        },
        "negative": {
            "slow",
            "late",
            "delayed",
        },
    },
    "shipping": {
        "positive": {
            "fast",
            "quick",
            "prompt",
            "early",
        },
        "negative": {
            "slow",
            "late",
            "delayed",
        },
    },
    "packaging": {
        "positive": {
            "good",
            "great",
            "excellent",
            "secure",
            "strong",
        },
        "negative": {
            "bad",
            "poor",
            "damaged",
            "weak",
            "terrible",
        },
    },
    "service": {
        "positive": {
            "good",
            "great",
            "excellent",
            "helpful",
            "fast",
        },
        "negative": {
            "bad",
            "poor",
            "slow",
            "terrible",
            "unhelpful",
        },
    },
}


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """
    Convert text to lowercase and normalise whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenise(text: str) -> List[str]:
    """
    Convert text into simple word tokens.
    """
    return re.findall(r"[a-z]+", text.lower())


# ---------------------------------------------------------------------------
# Sentence Handling
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """
    Split a review into simple sentence-like segments.

    This function is kept available for later integration and testing.
    """
    sentences = re.split(r"[.!?;]+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------------------------
# Aspect Location
# ---------------------------------------------------------------------------

def find_aspect_positions(
    tokens: List[str],
    aspect: str,
) -> List[int]:
    """
    Find the token positions where an aspect occurs.

    Supports both single-word and multi-word aspects.
    """
    aspect_tokens = tokenise(aspect)

    if not aspect_tokens:
        return []

    positions = []
    aspect_length = len(aspect_tokens)

    for index in range(
        len(tokens) - aspect_length + 1
    ):
        window = tokens[
            index:index + aspect_length
        ]

        if window == aspect_tokens:
            positions.append(index)

    return positions


# ---------------------------------------------------------------------------
# Local Context
# ---------------------------------------------------------------------------

def get_context_window(
    tokens: List[str],
    aspect: str,
    window_size: int = 5,
) -> List[str]:
    """
    Return a local context window around the first occurrence of an aspect.

    The local context prevents sentiment words that are very far away in a
    review from having unnecessary influence on the target aspect.
    """
    positions = find_aspect_positions(
        tokens=tokens,
        aspect=aspect,
    )

    if not positions:
        return []

    aspect_tokens = tokenise(aspect)

    aspect_start = positions[0]
    aspect_end = (
        aspect_start + len(aspect_tokens)
    )

    start = max(
        0,
        aspect_start - window_size,
    )

    end = min(
        len(tokens),
        aspect_end + window_size,
    )

    return tokens[start:end]


# ---------------------------------------------------------------------------
# Sentiment Scoring
# ---------------------------------------------------------------------------

def score_context(
    context: List[str],
    aspect: str,
) -> float:
    """
    Calculate the sentiment score associated with an aspect.

    The classifier gives priority to sentiment words closest to the target
    aspect. This prevents sentiment belonging to another aspect from
    incorrectly influencing the current aspect.

    Positive score  -> Positive sentiment
    Negative score  -> Negative sentiment
    Zero             -> Neutral sentiment
    """
    if not context:
        return 0.0

    aspect_name = aspect.lower()
    aspect_tokens = tokenise(aspect)

    # Find the target aspect inside the local context.
    aspect_positions = find_aspect_positions(
        tokens=context,
        aspect=aspect,
    )

    if not aspect_positions:
        return 0.0

    aspect_start = aspect_positions[0]

    aspect_end = (
        aspect_start + len(aspect_tokens) - 1
    )

    aspect_rules = ASPECT_CONTEXT_RULES.get(
        aspect_name
    )

    sentiment_candidates = []

    for index, word in enumerate(context):

        word_score = 0.0

        # ---------------------------------------------------------------
        # General sentiment lexicon
        # ---------------------------------------------------------------

        if word in POSITIVE_WORDS:
            word_score = 1.0

        elif word in NEGATIVE_WORDS:
            word_score = -1.0

        # ---------------------------------------------------------------
        # Aspect-specific sentiment rules
        # ---------------------------------------------------------------

        if aspect_rules:

            if word in aspect_rules["positive"]:
                word_score = 1.0

            elif word in aspect_rules["negative"]:
                word_score = -1.0

        # Ignore words with no sentiment evidence.
        if word_score == 0:
            continue

        # ---------------------------------------------------------------
        # Distance from the target aspect
        # ---------------------------------------------------------------

        if index < aspect_start:

            distance = (
                aspect_start - index
            )

        elif index > aspect_end:

            distance = (
                index - aspect_end
            )

        else:

            distance = 0

        # ---------------------------------------------------------------
        # Negation handling
        # ---------------------------------------------------------------

        previous_words = context[
            max(0, index - 3):index
        ]

        negation_found = any(
            previous_word in NEGATION_WORDS
            for previous_word in previous_words
        )

        if negation_found:
            word_score *= -1

        # ---------------------------------------------------------------
        # Intensifier and diminisher handling
        # ---------------------------------------------------------------

        multiplier = 1.0

        previous_word = (
            context[index - 1]
            if index > 0
            else ""
        )

        if previous_word in INTENSIFIERS:

            multiplier = INTENSIFIERS[
                previous_word
            ]

        elif previous_word in DIMINISHERS:

            multiplier = DIMINISHERS[
                previous_word
            ]

        adjusted_score = (
            word_score * multiplier
        )

        # ---------------------------------------------------------------
        # Store the sentiment evidence
        # ---------------------------------------------------------------

        sentiment_candidates.append(
            {
                "score": adjusted_score,
                "distance": distance,
                "aspect_specific": (
                    aspect_rules is not None
                    and (
                        word
                        in aspect_rules["positive"]
                        or word
                        in aspect_rules["negative"]
                    )
                ),
            }
        )

    # No sentiment evidence was found.
    if not sentiment_candidates:
        return 0.0

    # ---------------------------------------------------------------
    # Select the closest sentiment evidence.
    # ---------------------------------------------------------------

    nearest_distance = min(
        candidate["distance"]
        for candidate in sentiment_candidates
    )

    nearest_candidates = [
        candidate
        for candidate in sentiment_candidates
        if candidate["distance"]
        == nearest_distance
    ]

    # ---------------------------------------------------------------
    # If there is a tie, prefer aspect-specific evidence.
    # ---------------------------------------------------------------

    aspect_specific_candidates = [
        candidate
        for candidate in nearest_candidates
        if candidate["aspect_specific"]
    ]

    if aspect_specific_candidates:
        nearest_candidates = (
            aspect_specific_candidates
        )

    return sum(
        candidate["score"]
        for candidate in nearest_candidates
    )


# ---------------------------------------------------------------------------
# Score Classification
# ---------------------------------------------------------------------------

def classify_score(score: float) -> str:
    """
    Convert a numerical sentiment score into a sentiment label.
    """
    if score > 0:
        return "Positive"

    if score < 0:
        return "Negative"

    return "Neutral"


# ---------------------------------------------------------------------------
# Public Aspect Sentiment API
# ---------------------------------------------------------------------------

def classify_aspect_sentiment(
    review_text: str,
    aspect: str,
) -> Dict[str, object]:
    """
    Classify the sentiment associated with one aspect.

    Returns:
        aspect
        sentiment
        numerical score
        local context
    """
    text = normalise_text(review_text)

    tokens = tokenise(text)

    if not aspect:
        return {
            "aspect": "",
            "sentiment": "Neutral",
            "score": 0.0,
            "context": [],
        }

    context = get_context_window(
        tokens=tokens,
        aspect=aspect,
    )

    if not context:
        return {
            "aspect": aspect,
            "sentiment": "Neutral",
            "score": 0.0,
            "context": [],
        }

    score = score_context(
        context=context,
        aspect=aspect,
    )

    sentiment = classify_score(score)

    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "score": round(score, 3),
        "context": context,
    }


# ---------------------------------------------------------------------------
# Review-Level Aspect Sentiment API
# ---------------------------------------------------------------------------

def classify_review_aspects(
    review_text: str,
    aspects: List[str],
) -> List[Dict[str, object]]:
    """
    Classify sentiment for every aspect extracted from a review.
    """
    results = []

    for aspect in aspects:

        result = classify_aspect_sentiment(
            review_text=review_text,
            aspect=aspect,
        )

        results.append(result)

    return results