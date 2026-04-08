def grade_spam(pred: str, actual: str) -> int:
    """Grade spam classification. Returns 1 for correct, 0 for incorrect."""
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    return 1 if pred == actual else 0


def grade_category(pred: str, actual: str) -> int:
    """
    Grade category classification.
    - Correct match: 1
    - Wrong or invalid: 0
    """
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    if pred == actual:
        return 1
    return 0


def grade_reply(reply: str) -> int:
    """
    Grade reply quality.
    - Contains professional acknowledgment keywords: 1
    - General/wrong response: 0
    """
    reply = reply.lower()
    high_quality_keywords = [
        "thank", "noted", "understood", "received", "acknowledged",
        "will do", "on it", "i'll", "i will", "sure", "happy to"
    ]
    for keyword in high_quality_keywords:
        if keyword in reply:
            return 1
    return 0
