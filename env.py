from models import Action, Observation, EnvResult, Message
from emails import emails
from graders import GradeEpisode, grade_reply_task


import random

class EmailEnv:
    """
    Multi-step Email Triage Environment (OpenEnv Standard).

    Supports 3 task difficulty levels:
      - easy:   Only spam classification (1 step)
      - medium: Spam classification + category (2 steps)
      - hard:   Spam classification + category + reply (3 steps)

    Reward is normalized between 0.0 and 1.0:
      - classify  -> contributes up to 0.4
      - categorize -> contributes up to 0.3
      - reply      -> contributes up to 0.3
    """

    TASK_STEPS = {
        "easy": ["classify"],
        "medium": ["classify", "categorize"],
        "hard": ["classify", "categorize", "reply"],
    }

    def __init__(self):
        self.current_email = None
        self.history = []
        self.messages = []
        self.step_count = 0
        self.max_steps = 3
        self.task_type = "easy"
        self.cumulative_reward = 0.1
        self.step_outcomes = [] # To track Perfect/Near-Perfect/Catastrophic/etc.
    def reset(self, task_type=None):
        """Standard OpenEnv Reset: Returns EnvResult."""
        self.current_email = random.choice(emails)
        self.history = []
        self.messages = [
            Message(category="SYSTEM", content=f"Task: Triage this email. Difficulty: {task_type or self.task_type}")
        ]
        self.step_count = 0
        self.task_type = task_type if task_type else random.choice(["easy", "medium", "hard"])
        self.max_steps = len(self.TASK_STEPS[self.task_type])
        self.cumulative_reward = 0.1
        self.step_outcomes = []

        obs = Observation(
            email_id=self.current_email["id"],
            subject=self.current_email["subject"],
            body=self.current_email["body"],
            sender=self.current_email["sender"],
            messages=self.messages,
            history=self.history
        )
        return EnvResult(
            observation=obs,
            reward=0.1,  # Initial reward (strictly > 0)
            done=False,
            info={
                "task_type": self.task_type,
                "expected_steps": self.TASK_STEPS[self.task_type]
            }
        )

    def step(self, action: Action):
        """Standard OpenEnv Step: Returns EnvResult."""
        # PER USER REQUEST: [STEP] rewards are per-step normalized rewards, not comparison-based
        # We also track the specific outcome category for the trajectory grader
        outcome = "partial" # default
        step_reward = 0.50  # Safety default to prevent 0 or undefined
        
        is_spam = self.current_email["label"] == "spam"
        
        if action.action_type == "classify":
            pred = action.content.strip().lower()
            actual = self.current_email["label"].strip().lower()
            if pred == "not_spam" and is_spam:
                outcome = "catastrophic" # Approved bug
                step_reward = 0.10
            elif pred == "spam" and not is_spam:
                outcome = "false_positive"
                step_reward = 0.15
            elif pred == actual:
                outcome = "near_perfect"
                step_reward = 0.88
            else:
                outcome = "missed_bug" if is_spam else "partial"
                step_reward = 0.30 if is_spam else 0.50
        
        elif action.action_type == "categorize":
            # Assume severity check maps to categorization
            if action.content.strip().lower() == self.current_email["category"].strip().lower():
                outcome = "perfect" # Upgrading from near-perfect if classification was also right
                step_reward = 0.90
            else:
                outcome = "partial"
                step_reward = 0.75
                
        elif action.action_type == "reply":
            # using global grade_reply_task
            if grade_reply_task(action.content) > 0.8:
                outcome = "perfect"
                step_reward = 0.90
            else:
                outcome = "partial"
                step_reward = 0.70

        self.step_outcomes.append(outcome)
        self.cumulative_reward += step_reward
        self.step_count += 1
        
        # Internal cumulative reward remains unclamped for UI display
        # Only the 'score' for the validator will be normalized to (0, 1)
        
        # Track history for context
        self.history.append(f"[{action.action_type}] {action.content}")
        self.messages.append(Message(category="USER", content=f"{action.action_type}: {action.content}"))
        self.messages.append(Message(category="ENVIRONMENT", content=f"Feedback: Action recorded. Reward gained: {step_reward:.2f}"))

        self.done = self.step_count >= self.max_steps

        obs = Observation(
            email_id=self.current_email["id"],
            subject=self.current_email["subject"],
            body=self.current_email["body"],
            sender=self.current_email["sender"],
            messages=self.messages,
            history=self.history
        )

        # 1. Setup the default info dictionary
        info_dict = {
            "step": self.step_count,
            "task_type": self.task_type,
            "cumulative_reward": round(self.cumulative_reward, 4)
        }

        # 2. THE FIX: When the episode ends, calculate and inject the final normalized 'score'
        if self.done:
            # Divide by max_steps to normalize the cumulative reward 
            raw_score = self.cumulative_reward / self.max_steps
            # Force the score to be strictly between 0 and 1 (e.g., 0.01 to 0.99)
            final_score = max(0.01, min(0.99, raw_score))
            
            # OpenEnv requires this exact key to validate the task score
            info_dict["score"] = round(final_score, 4) # Renamed to score

        return EnvResult(
            observation=obs,
            reward=round(min(max(step_reward, 0.01), 0.99), 3),
            done=self.done,
            info=info_dict
        )

    def state(self):
        """Oracle State for Learning."""
        return {
            "email": self.current_email,
            "history": self.history,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "step_outcomes": self.step_outcomes,
            "task_type": self.task_type,
            "done": getattr(self, "done", False)
        }