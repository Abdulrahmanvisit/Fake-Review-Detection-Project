import html
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

CLAUSE_BOUNDARIES = {
    ".",
    "!",
    "?",
    ";",
    ",",
    "but",
    "however",
    "although",
    "though",
    "yet",
    "whereas",
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
    "battery": {
        "positive": {
            "good",
            "excellent",
            "great",
            "long",
            "last",
            "lasts",
            "lasting",
            "work",
            "works",
            "worked",
            "working",
        },
        "negative": {
            "short",
            "poor",
            "weak",
            "stopped",
            "failed",
            "dead",
            "drains",
            "drained",
            "dies",
            "dying",
        },
    },
    "battery life": {
        "positive": {
            "good",
            "excellent",
            "great",
            "long",
            "lasts",
            "lasting",
        },
        "negative": {
            "short",
            "poor",
            "weak",
            "brief",
            "drains",
            "drained",
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

    text = html.unescape(text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenise(text: str) -> List[str]:
    """
    Convert text into Unicode-aware word and clause-boundary tokens.
    """
    text = normalise_text(text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"\b([a-z]+)n't\b", r"\1 not", text)

    return re.findall(
        r"[^\W\d_]+|[.!?;,]",
        text,
        flags=re.UNICODE,
    )


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
    aspect_end = aspect_start + len(aspect_tokens)

    clause_start = 0
    for index, token in enumerate(tokens[:aspect_start]):
        if token in CLAUSE_BOUNDARIES:
            clause_start = index + 1

    clause_end = len(tokens)
    for index in range(aspect_end, len(tokens)):
        if tokens[index] in CLAUSE_BOUNDARIES:
            clause_end = index
            break

    start = max(
        clause_start,
        aspect_start - window_size,
    )

    end = min(
        clause_end,
        aspect_end + window_size,
    )

    return tokens[start:end]


# ---------------------------------------------------------------------------
# Sentiment Scoring
# ---------------------------------------------------------------------------

def get_context_windows(
    tokens: List[str],
    aspect: str,
    window_size: int = 5,
) -> List[List[str]]:
    """Return an independent local context for each aspect occurrence."""
    positions = find_aspect_positions(tokens, aspect)
    aspect_tokens = tokenise(aspect)
    contexts = []

    for aspect_start in positions:
        aspect_end = aspect_start + len(aspect_tokens)
        clause_start = 0
        for index, token in enumerate(tokens[:aspect_start]):
            if token in CLAUSE_BOUNDARIES:
                clause_start = index + 1

        clause_end = len(tokens)
        for index in range(aspect_end, len(tokens)):
            if tokens[index] in CLAUSE_BOUNDARIES:
                clause_end = index
                break

        start = max(clause_start, aspect_start - window_size)
        end = min(clause_end, aspect_end + window_size)
        contexts.append(tokens[start:end])

    return contexts


def score_context(context: List[str], aspect: str) -> float:
    """Score the closest sentiment evidence within one aspect context."""
    if not context:
        return 0.0

    aspect_name = aspect.lower()
    aspect_tokens = tokenise(aspect)
    aspect_positions = find_aspect_positions(context, aspect)
    if not aspect_positions:
        return 0.0

    aspect_start = aspect_positions[0]
    aspect_end = aspect_start + len(aspect_tokens) - 1
    aspect_rules = ASPECT_CONTEXT_RULES.get(aspect_name)
    if aspect_rules is None and " " in aspect_name:
        aspect_rules = ASPECT_CONTEXT_RULES.get(aspect_name.rsplit(" ", 1)[-1])

    sentiment_candidates = []
    for index, word in enumerate(context):
        word_score = 0.0
        if word in POSITIVE_WORDS:
            word_score = 1.0
        elif word in NEGATIVE_WORDS:
            word_score = -1.0

        if aspect_rules:
            if word in aspect_rules["positive"]:
                word_score = 1.0
            elif word in aspect_rules["negative"]:
                word_score = -1.0

        if word_score == 0:
            continue

        if index < aspect_start:
            distance = aspect_start - index
        elif index > aspect_end:
            distance = index - aspect_end
        else:
            distance = 0

        previous_words = context[max(0, index - 3):index]
        if any(previous_word in NEGATION_WORDS for previous_word in previous_words):
            word_score *= -1

        previous_word = context[index - 1] if index > 0 else ""
        if previous_word in INTENSIFIERS:
            multiplier = INTENSIFIERS[previous_word]
        elif previous_word in DIMINISHERS:
            multiplier = DIMINISHERS[previous_word]
        else:
            multiplier = 1.0

        sentiment_candidates.append(
            {
                "score": word_score * multiplier,
                "distance": distance,
                "aspect_specific": (
                    aspect_rules is not None
                    and word in (aspect_rules["positive"] | aspect_rules["negative"])
                ),
            }
        )

    if not sentiment_candidates:
        return 0.0

    nearest_distance = min(
        candidate["distance"] for candidate in sentiment_candidates
    )
    nearest_candidates = [
        candidate
        for candidate in sentiment_candidates
        if candidate["distance"] == nearest_distance
    ]
    aspect_specific_candidates = [
        candidate
        for candidate in nearest_candidates
        if candidate["aspect_specific"]
    ]
    if aspect_specific_candidates:
        nearest_candidates = aspect_specific_candidates

    return sum(candidate["score"] for candidate in nearest_candidates)


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

    contexts = get_context_windows(tokens, aspect)
    if not contexts:
        return {
            "aspect": aspect,
            "sentiment": "Neutral",
            "score": 0.0,
            "context": [],
        }

    occurrence_scores = [
        score_context(context, aspect)
        for context in contexts
    ]
    score = sum(occurrence_scores) / len(occurrence_scores)

    return {
        "aspect": aspect,
        "sentiment": classify_score(score),
        "score": round(score, 3),
        "context": contexts[0],
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