from fastapi import FastAPI, Request
import gradio as gr
from env import EmailEnv
from models import Action, EnvResult
import uvicorn
from app import demo # Import the Gradio demo object

app = FastAPI(title="Email Triage RL Environment - OpenEnv API")
env = EmailEnv()

@app.get("/health")
def health():
    return {"status": "ok", "env": "email_triage"}

@app.post("/reset")
def reset():
    """Standard OpenEnv Reset Endpoint."""
    result = env.reset()
    return result

@app.post("/step")
def step(action: Action):
    """Standard OpenEnv Step Endpoint."""
    result = env.step(action)
    return result

@app.get("/state")
def get_state():
    """Standard OpenEnv State Endpoint."""
    return env.state()

# Mount the Gradio UI at /ui or /
# On Hugging Face Spaces, users usually expect the UI at the root.
# We can mount it at / and it will handle the Gradio routes.
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
