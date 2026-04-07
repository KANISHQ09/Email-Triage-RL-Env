from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from env import EmailEnv
from models import Action, EnvResult
import uvicorn
from ui import demo # Import the Gradio demo object

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
    return RedirectResponse(url="/ui")

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

# Mount the Gradio UI at /ui to avoid route collision with the REST API at root
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
