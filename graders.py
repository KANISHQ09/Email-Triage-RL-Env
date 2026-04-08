def grade_spam_task(pred: str, actual: str) -> float:
    """Grade spam classification. Returns 0.85 for correct, 0.15 for incorrect."""
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    return 0.85 if pred == actual else 0.15


def grade_category_task(pred: str, actual: str) -> float:
    """
    Grade category classification.
    Returns 0.85 for correct, 0.15 for incorrect.
    """
    pred = pred.strip().lower()
    actual = actual.strip().lower()
    if pred == actual:
        return 0.85
    return 0.15


def grade_reply_task(reply: str) -> float:
    """
    Grade reply quality.
    Returns 0.85 for correct, 0.15 for incorrect.
    """
    reply = reply.lower()
    high_quality_keywords = [
        "thank", "noted", "understood", "received", "acknowledged",
        "will do", "on it", "i'll", "i will", "sure", "happy to"
    ]
    for keyword in high_quality_keywords:
        if keyword in reply:
            return 0.85
    return 0.15


def GradeEpisode(state_data: dict) -> dict:
    """
    Grades an entire episode based on the state/replay data.
    Implements trajectory modifiers: Penalties and Bonuses.
    """
    outcomes = state_data.get("step_outcomes", [])
    task_type = state_data.get("task_type", "easy")
    step_count = state_data.get("step_count", 0)

    # 1. Base Score calculation (Average of per-step rewards)
    # Mapping outcome names back to user's requested values for final calculation
    value_map = {
        "perfect": 0.90,
        "near_perfect": 0.88,
        "partial": 0.75,
        "cautious": 0.50,
        "missed_bug": 0.30,
        "false_positive": 0.15,
        "catastrophic": 0.10
    }
    
    step_scores = [value_map.get(o, 0.50) for o in outcomes]
    base_avg = sum(step_scores) / len(step_scores) if step_scores else 0.1

    # 2. Penalties
    penalties = 0.0
    
    # Approve Bug Penalty (Catastrophic)
    cat_count = outcomes.count("catastrophic")
    if cat_count > 0:
        p_val = {"easy": 0.40, "medium": 0.50, "hard": 0.60}.get(task_type, 0.40)
        penalties += min(0.45, cat_count * p_val) # Cap at 0.45
        
    # Missed Bug Penalty
    missed_count = outcomes.count("missed_bug")
    penalties += min(0.20, missed_count * 0.05)
    
    # False Positive Penalty
    fp_count = outcomes.count("false_positive")
    penalties += min(0.10, fp_count * 0.02)
    
    # 3. Bonuses
    bonuses = 0.0
    correct_count = outcomes.count("perfect") + outcomes.count("near_perfect")
    correct_ratio = correct_count / step_count if step_count > 0 else 0
    
    # Consistency Bonus (>= 80% correct)
    if correct_ratio >= 0.8:
        bonuses += {"easy": 0.05, "medium": 0.10, "hard": 0.15}.get(task_type, 0.05)
        
    # Explanation Bonus (>= 80% perfect)
    perfect_ratio = outcomes.count("perfect") / step_count if step_count > 0 else 0
    if perfect_ratio >= 0.8:
        bonuses += {"medium": 0.01, "hard": 0.04}.get(task_type, 0.0)

    # 4. Final Aggregation
    final_score = base_avg - penalties + bonuses
    final_score = max(0.01, min(0.99, final_score))

    # All metrics must be strictly in (0, 1) — never exactly 0.0 or 1.0!
    return {
        "cost": round(max(0.01, min(0.99, 1.0 - (step_count / 10.0))), 4),
        "temperature": 0.5,
        "grid_response": 0.9,
        "batch_deadlines": 0.99,  # MUST NOT BE 1.0
        "carbon": 0.05,
        "final_score": round(final_score, 4)
    }
