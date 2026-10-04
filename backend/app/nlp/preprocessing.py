import re
from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Words that should never be removed because they can
# change the meaning of a sentiment.
NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "hardly",
    "barely",
    "scarcely",
}


def remove_html(text: str) -> str:
    """Remove HTML tags such as <br /> from review text."""

    return re.sub(r"<[^>]+>", " ", text)


def normalise_whitespace(text: str) -> str:
    """Replace repeated whitespace with a single space."""

    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Perform conservative text cleaning while preserving
    words, Unicode characters, negation and useful punctuation.
    """

    if not isinstance(text, str):
        return ""

    # Remove HTML tags.
    text = remove_html(text)

    # Convert text to lowercase.
    text = text.lower()

    # Keep Unicode word characters, whitespace and
    # punctuation useful for sentiment analysis.
    text = re.sub(r"[^\w\s'!?.,-]", " ", text, flags=re.UNICODE)

    # Normalise repeated whitespace.
    text = normalise_whitespace(text)

    return text


def tokenise_text(text: str) -> List[str]:
    """Tokenise cleaned review text."""

    return word_tokenize(text)


def preserve_negations(tokens: List[str]) -> List[str]:
    """
    Join common contraction forms so that negation
    remains explicit for sentiment analysis.

    Examples:
        does + n't -> does_not
        do + n't    -> do_not
        is + n't    -> is_not
        n't         -> not
    """

    processed_tokens = []

    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "n't":
            processed_tokens.append("not")

        elif (
            token in {"do", "does", "did", "is", "are", "was", "were",
                      "have", "has", "had", "can", "could", "will",
                      "would", "should", "must", "might", "may"}
            and i + 1 < len(tokens)
            and tokens[i + 1] == "n't"
        ):
            processed_tokens.append(f"{token}_not")
            i += 1

        else:
            processed_tokens.append(token)

        i += 1

    return processed_tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common English stopwords while preserving
    important negation words.
    """

    stop_words = set(stopwords.words("english"))

    # Keep negation words because they can change sentiment.
    stop_words -= NEGATION_WORDS

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


def preprocess_review(text: str) -> dict:
    """
    Run the complete preprocessing pipeline on one review.
    """

    cleaned_text = clean_text(text)

    tokens = tokenise_text(cleaned_text)

    tokens = preserve_negations(tokens)

    filtered_tokens = remove_stopwords(tokens)

    processed_text = " ".join(filtered_tokens)

    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "processed_text": processed_text,
    }


if __name__ == "__main__":
    sample_review = (
        "The battery is not good, but the delivery was excellent! "
        "<br /> I would buy it again. The product doesn't disappoint."
    )

    result = preprocess_review(sample_review)

    print("Original:")
    print(sample_review)

    print("\nCleaned text:")
    print(result["cleaned_text"])

    print("\nTokens:")
    print(result["tokens"])

    print("\nFiltered tokens:")
    print(result["filtered_tokens"])

    print("\nProcessed text:")
    print(result["processed_text"])