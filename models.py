from pydantic import BaseModel
from typing import List


class Action(BaseModel):
    action_type: str   # classify / categorize / reply
    content: str       # label or reply text


class Observation(BaseModel):
    email_id: int
    subject: str
    body: str
    sender: str
    history: List[str]
