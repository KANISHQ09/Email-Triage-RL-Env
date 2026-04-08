# graders.py
# Reference-based grading logic for Email Triage Environment.
# All scores are strictly in (0.01, 0.99) with 3-decimal precision.

def GradeEpisode(state: dict) -> dict:
    """
    Main router for state-based grading.
    Returns a dictionary of metrics strictly in (0.01, 0.99).
    """
    history = state.get("history", [])
    step_outcomes = state.get("step_outcomes", [])
    task_type = state.get("task_type", "easy")
    
    # Calculate raw scores based on trajectory outcomes
    # value_map: perfect=0.9, near_perfect=0.8, mixed=0.5, fail=0.2, catastrophic=0.05
    scores = []
    for outcome in step_outcomes:
        if outcome == "perfect": scores.append(0.90)
        elif outcome == "near_perfect": scores.append(0.80)
        elif outcome == "partial": scores.append(0.50)
        elif outcome == "missed_bug": scores.append(0.25)
        elif outcome == "false_positive": scores.append(0.15)
        elif outcome == "catastrophic": scores.append(0.05)
        else: scores.append(0.10) # default small non-zero
        
    base_avg = sum(scores) / len(scores) if scores else 0.50
    
    # Trajectory Modifiers (Penalties/Bonuses)
    penalties = 0.0
    bonuses = 0.0
    
    # "Approve Bug" Penalty (Catastrophic)
    catastrophic_count = step_outcomes.count("catastrophic")
    if catastrophic_count > 0:
        penalties += min(0.40, catastrophic_count * 0.40)
        
    # Consistency Bonus
    if len(step_outcomes) >= 2 and step_outcomes.count("perfect") + step_outcomes.count("near_perfect") >= len(step_outcomes) * 0.8:
        bonuses += 0.10
        
    raw_score = base_avg - penalties + bonuses
    
    # OpenEnv requirement: scores must be strictly in (0.0, 1.0)
    # Using the reference pattern: round(min(max(raw_score, 0.01), 0.99), 3)
    return {
        "score":           round(min(max(raw_score, 0.01), 0.99), 3),
        "cost":            round(min(max(1.0 - (len(step_outcomes)/10.0), 0.01), 0.99), 3),
        "temperature":     0.5,
        "grid_response":   0.9,
        "batch_deadlines": 0.99,
        "carbon":          0.05
    }

# ---------------------------------------------------------------------------
# Discrete Task-Specific Graders (Discovery Aliases)
# ---------------------------------------------------------------------------

def GradeSpam(state: dict) -> dict:
    """Explicit grader for Spam Detection tasks."""
    return GradeEpisode(state)

def GradeCategory(state: dict) -> dict:
    """Explicit grader for Category Classification tasks."""
    return GradeEpisode(state)

def GradeFull(state: dict) -> dict:
    """Explicit grader for Full Pipeline tasks."""
    return GradeEpisode(state)


# Legacy Logic Helpers (for step rewards)
def grade_reply_task(content: str) -> float:
    # Basic quality check for intermediate rewards
    if not content or len(content) < 10: return 0.2
    if "thank" in content.lower() or "noted" in content.lower(): return 0.9
    return 0.5
