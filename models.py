from pydantic import BaseModel
from typing import List, Optional, Any, Union


class Action(BaseModel):
    action_type: str   # classify / categorize / reply
    content: str       # label or reply text


class Message(BaseModel):
    category: str      # e.g., "SYSTEM", "USER", "ENVIRONMENT"
    content: str
    sender: Optional[str] = None


class Observation(BaseModel):
    email_id: int
    subject: str
    body: str
    sender: str
    messages: List[Message] = []
    history: List[str] = [] # Legacy field for compatibility


class EnvResult(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: dict = {}
