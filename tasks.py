def grade_spam(pred: str, actual: str) -> float:
    """Grade spam classification. Returns 0.99 for correct, 0.01 for incorrect."""
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    return 0.99 if pred == actual else 0.01


def grade_category(pred: str, actual: str) -> float:
    """
    Grade category classification.
    - Correct match: 0.99
    - Wrong or invalid: 0.01
    """
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    if pred == actual:
        return 0.99
    return 0.01


def grade_reply(reply: str) -> float:
    """
    Grade reply quality.
    - Contains professional acknowledgment keywords: 0.99
    - General/wrong response: 0.01
    """
    reply = reply.lower()
    high_quality_keywords = [
        "thank", "noted", "understood", "received", "acknowledged",
        "will do", "on it", "i'll", "i will", "sure", "happy to"
    ]
    for keyword in high_quality_keywords:
        if keyword in reply:
            return 0.99
    return 0.01