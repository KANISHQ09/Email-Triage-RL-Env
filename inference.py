import os
import sys
import textwrap
from typing import List, Optional
from openai import OpenAI

from env import EmailEnv
from models import Action

# --- Mandatory Variables (Guidelines v2) ---
# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Benchmarking Identifiers
TASK_NAME = "email_triage"
BENCHMARK = "email_env"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


# --- LLM Helper ---
def get_llm_response(prompt: str) -> str:
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return res.choices[0].message.content.strip()


# --- Prompt Engineering ---
def build_classify_prompt(obs) -> str:
    return textwrap.dedent(f"""
        You are an expert email assistant.
        Sender: {obs.sender}
        Subject: {obs.subject}
        Body: {obs.body}
        Classify as 'spam' or 'not_spam'. Respond with exactly one word.
    """).strip()

def build_categorize_prompt(obs) -> str:
    return textwrap.dedent(f"""
        You are an expert email assistant.
        Sender: {obs.sender}
        Subject: {obs.subject}
        Body: {obs.body}
        Categorize into: 'work', 'personal', 'promotion'. Respond with exactly one word.
    """).strip()

def build_full_prompt(obs) -> str:
    return textwrap.dedent(f"""
        You are an intelligent email triage assistant.
        Sender: {obs.sender}
        Subject: {obs.subject}
        Body: {obs.body}
        Respond in EXACTLY this format:
        classify: <spam or not_spam>
        categorize: <work or personal or promotion>
        reply: <your reply here>
    """).strip()

def parse_full_response(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result


# --- Main Inference Loop ---
def main():
    env = EmailEnv()
    rewards: List[float] = []
    step_count = 0
    success = False
    error_msg = "null"

    # 1. [START] line
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        # Reset Env
        result = env.reset()
        obs = result.observation
        task_type = result.info.get("task_type", "easy")
        
        # Decide Actions
        if task_type == "hard":
            raw_response = get_llm_response(build_full_prompt(obs))
            parsed = parse_full_response(raw_response)
            actions = [
                ("classify", parsed.get("classify", "not_spam")),
                ("categorize", parsed.get("categorize", "work")),
                ("reply", parsed.get("reply", "Thank you."))
            ]
        elif task_type == "medium":
            actions = [
                ("classify", get_llm_response(build_classify_prompt(obs))),
                ("categorize", get_llm_response(build_categorize_prompt(obs)))
            ]
        else:
            actions = [
                ("classify", get_llm_response(build_classify_prompt(obs)))
            ]

        # Execute
        for action_type, content in actions:
            step_count += 1
            action_obj = Action(action_type=action_type, content=content)
            
            res = env.step(action_obj)
            reward = res.reward
            done = res.done
            rewards.append(reward)
            
            # 2. [STEP] line (Exactly one space)
            print(f"[STEP] step={step_count} action={action_type} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
            if done:
                break
        
        # Simple success heuristic: non-zero reward
        success = sum(rewards) >= 0.1

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        success = False

    finally:
        # 3. [END] line (Exactly one space, remove 'score')
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")
        if hasattr(env, "close"):
            env.close()

if __name__ == "__main__":
    main()
