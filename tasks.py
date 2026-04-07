def grade_spam(pred: str, actual: str) -> float:
    """Grade spam classification. Returns 1.0 for correct, 0.0 for incorrect."""
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    return 1.0 if pred == actual else 0.0


def grade_category(pred: str, actual: str) -> float:
    """
    Grade category classification.
    - Correct match: 1.0
    - Valid category but wrong: 0.5
    - Invalid category: 0.0
    """
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    valid_categories = {"work", "personal", "promotion"}

    if pred == actual:
        return 1.0
    elif pred in valid_categories:
        return 0.5
    return 0.0


def grade_reply(reply: str) -> float:
    """
    Grade reply quality.
    - Contains professional acknowledgment keywords: 1.0
    - General response: 0.5
    """
    reply = reply.lower()
    high_quality_keywords = [
        "thank", "noted", "understood", "received", "acknowledged",
        "will do", "on it", "i'll", "i will", "sure", "happy to"
    ]
    for keyword in high_quality_keywords:
        if keyword in reply:
            return 1.0
    return 0.5
