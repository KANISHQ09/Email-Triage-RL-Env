import random
from models import Action, Observation, EnvResult, Message
from emails import emails
from tasks import grade_spam, grade_category, grade_reply


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
        self.cumulative_reward = 0.0

    def reset(self):
        """Standard OpenEnv Reset: Returns EnvResult."""
        self.current_email = random.choice(emails)
        self.history = []
        self.messages = [
            Message(category="SYSTEM", content=f"Task: Triage this email. Difficulty: {self.task_type}")
        ]
        self.step_count = 0
        self.task_type = random.choice(["easy", "medium", "hard"])
        self.max_steps = len(self.TASK_STEPS[self.task_type])
        self.cumulative_reward = 0.0

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
            reward=0.0,
            done=False,
            info={
                "task_type": self.task_type,
                "expected_steps": self.TASK_STEPS[self.task_type]
            }
        )

    def step(self, action: Action):
        """Standard OpenEnv Step: Returns EnvResult."""
        reward = 0.0

        # --- Weighted reward shaping ---
        if action.action_type == "classify":
            reward += 0.4 * grade_spam(action.content, self.current_email["label"])

        elif action.action_type == "categorize":
            reward += 0.3 * grade_category(action.content, self.current_email["category"])

        elif action.action_type == "reply":
            reward += 0.3 * grade_reply(action.content)

        self.cumulative_reward += reward
        self.step_count += 1
        
        # Track history for context
        self.history.append(f"[{action.action_type}] {action.content}")
        self.messages.append(Message(category="USER", content=f"{action.action_type}: {action.content}"))
        self.messages.append(Message(category="ENVIRONMENT", content=f"Feedback: Action recorded. Reward gained: {reward:.2f}"))

        self.done = self.step_count >= self.max_steps

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
            reward=round(reward, 4),
            done=self.done,
            info={
                "step": self.step_count,
                "task_type": self.task_type,
                "cumulative_reward": round(self.cumulative_reward, 4)
            }
        )

    def state(self):
        """Oracle State for Learning."""
        return {
            "email": self.current_email,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "done": getattr(self, "done", False)
        }
