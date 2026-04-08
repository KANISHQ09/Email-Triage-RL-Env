from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import gradio as gr
from env import EmailEnv
from models import Action, EnvResult
from tasks import grade_spam, grade_category, grade_reply
import uvicorn
from ui import demo  # Import the Gradio demo object

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
        "action_type": "classify",
        "grader": "tasks.grade_spam",
        "score": 0.4,
    },
    {
        "id": "category_classification",
        "name": "Category Classification",
        "difficulty": "medium",
        "description": "Classify + categorize email (work / personal / promotion)",
        "action_types": ["classify", "categorize"],
        "grader": "tasks.grade_category",
        "score": 0.7,
    },
    {
        "id": "full_decision",
        "name": "Full Decision",
        "difficulty": "hard",
        "description": "Full pipeline — classify, categorize, and reply",
        "action_types": ["classify", "categorize", "reply"],
        "grader": "tasks.grade_reply",
        "score": 0.99,
    },
]


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their grader definitions."""
    return {"tasks": TASKS}


class GradeRequest(BaseModel):
    task_name: str
    prediction: str
    ground_truth: Optional[str] = ""


@app.post("/grade")
def grade(request: GradeRequest):
    """
    Run the grader for a given task and return a score strictly in (0, 1).
    Scores are always in the open interval — never 0.0 or 1.0.
    """
    task_name = request.task_name
    prediction = request.prediction
    ground_truth = request.ground_truth

    if task_name == "spam_detection":
        score = grade_spam(prediction, ground_truth)
    elif task_name == "category_classification":
        score = grade_category(prediction, ground_truth)
    elif task_name == "full_decision":
        score = grade_reply(prediction)
    else:
        return {"error": f"Unknown task: {task_name}", "score": None}

    # Clamp to strict open interval (0, 1) — never exactly 0.0 or 1.0
    score = max(0.01, min(0.99, float(score)))
    return {"task": task_name, "score": score}

# Mount the Gradio UI at /ui to avoid route collision with the REST API at root
app = gr.mount_gradio_app(app, demo, path="/ui")

def main():
    """Entry point for openenv multi-mode deployment."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, proxy_headers=True, forwarded_allow_ips="*")

if __name__ == "__main__":
    main()
