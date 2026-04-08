from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import gradio as gr
from env import EmailEnv
from graders import GradeEpisode
from models import Action, EnvResult, EpisodeGrade
import uvicorn
from ui import demo

app = FastAPI(title="Email Triage RL Environment - OpenEnv API")

# Add CORS Middleware to allow automated testers from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailEnv()

@app.get("/health")
def health():
    return {"status": "ok", "env": "email_triage"}

@app.get("/")
def root_redirect():
    """Redirect human visitors to the UI."""
    return RedirectResponse(url="./ui/")

# Support both GET and POST for reset to prevent "Method Not Allowed" during diagnostic checks
@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    """Standard OpenEnv Reset Endpoint."""
    result = env.reset()
    return result

# Support both GET and POST for step
@app.api_route("/step", methods=["GET", "POST"])
def step(action: Action = None, request: Request = None):
    """Standard OpenEnv Step Endpoint."""
    # If it's a GET or empty POST, we might need to handle it gracefully
    if action is None:
        return {"error": "Missing action in request body"}
    result = env.step(action)
    return result

@app.get("/state")
def get_state():
    """Standard OpenEnv State Endpoint."""
    return env.state()


# ── Task & Grader Discovery ──────────────────────────────────────────────────

TASKS = [
    {
        "id": "spam_detection",
        "name": "Spam Detection",
        "difficulty": "easy",
        "description": "Classify email as spam or not_spam",
        "action_types": ["classify"],
        "grader": "GradeSpam",
        "score": 0.85,
    },
    {
        "id": "category_classification",
        "name": "Category Classification",
        "difficulty": "medium",
        "description": "Classify + categorize email",
        "action_types": ["classify", "categorize"],
        "grader": "GradeCategory",
        "score": 0.85,
    },
    {
        "id": "full_decision",
        "name": "Full Decision",
        "difficulty": "hard",
        "description": "Full pipeline — classify, categorize, and reply",
        "action_types": ["classify", "categorize", "reply"],
        "grader": "GradeFull",
        "score": 0.85,
    },
]


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their grader definitions."""
    return {"tasks": TASKS}


@app.get("/graders")
def list_graders():
    """List all available graders for discovery."""
    return {
        "graders": [
            {"id": "GradeSpam",     "name": "Spam Detection Grader", "type": "state_based"},
            {"id": "GradeCategory", "name": "Category Grader",       "type": "state_based"},
            {"id": "GradeFull",     "name": "Full Pipeline Grader", "type": "state_based"},
        ]
    }


@app.post("/grade", response_model=EpisodeGrade)
def grade_episode(state_data: dict):
    """
    Grades an episode based on the full state/replay data.
    Ensures 'score' is returned for manifest compliance.
    """
    grade_dict = GradeEpisode(state_data)
    # The models handle the rename of final_score -> score
    return EpisodeGrade(**grade_dict)

# Mount the Gradio UI at /ui to avoid route collision with the REST API at root
app = gr.mount_gradio_app(app, demo, path="/ui")

def main():
    """Entry point for openenv multi-mode deployment."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, proxy_headers=True, forwarded_allow_ips="*")

if __name__ == "__main__":
    main()
