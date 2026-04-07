import os
import re
import sys
import textwrap
from typing import List, Optional
from openai import OpenAI

from env import EmailEnv
from models import Action

# --- Mandatory Variables ---
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") 

# Benchmarking Identifiers
TASK_NAME = os.getenv("EMAIL_TASK", "email_triage")
BENCHMARK = os.getenv("EMAIL_BENCHMARK", "email_env_v1")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is missing.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# --- LLM Helper ---
def get_llm_response(prompt: str) -> str:
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 # Keep it deterministic for triage
    )
    return res.choices[0].message.content.strip()


# --- Prompt Engineering ---
def build_classify_prompt(obs) -> str:
    return textwrap.dedent(f"""
        You are an expert email assistant.
        
        Email:
        Sender: {obs.sender}
        Subject: {obs.subject}
        Body: {obs.body}
        
        Classify this email as 'spam' or 'not_spam'.
        Respond with exactly one word.
    """).strip()

def build_categorize_prompt(obs) -> str:
    return textwrap.dedent(f"""
        You are an expert email assistant.
        
        Email:
        Sender: {obs.sender}
        Subject: {obs.subject}
        Body: {obs.body}
        
        Categorize this email into one of: 'work', 'personal', 'promotion'.
        Respond with exactly one word.
    """).strip()

def build_full_prompt(obs) -> str:
    return textwrap.dedent(f"""
        You are an intelligent email triage assistant. Analyze this email and provide detection, categorization, and a reply.

        Email:
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


# --- Main Inference ---
def main():
    env = EmailEnv()
    rewards = []
    step_count = 0
    success = False
    error_msg = "null"

    # Start Logging
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        # Reset Env
        result = env.reset()
        obs = result.observation
        task_type = result.info["task_type"]
        
        # Determine steps based on task_type
        if task_type == "hard":
            raw_response = get_llm_response(build_full_prompt(obs))
            parsed = parse_full_response(raw_response)
            actions = [
                ("classify", parsed.get("classify", "not_spam")),
                ("categorize", parsed.get("categorize", "work")),
                ("reply", parsed.get("reply", "Understood. Thank you."))
            ]
        elif task_type == "medium":
            actions = [
                ("classify", get_llm_response(build_classify_prompt(obs))),
                ("categorize", get_llm_response(build_categorize_prompt(obs)))
            ]
        else: # easy
            actions = [
                ("classify", get_llm_response(build_classify_prompt(obs)))
            ]

        # Execute Actions
        for action_type, content in actions:
            step_count += 1
            action = Action(action_type=action_type, content=content)
            
            # OpenEnv Step
            result = env.step(action)
            
            reward = result.reward
            done = result.done
            rewards.append(reward)
            
            # STEP LOGGING (Mandatory Format: Single Space)
            print(f"[STEP] step={step_count} action={action_type} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
            if done:
                break
        
        total_score = sum(rewards)
        success = total_score >= 0.1 # Mirroring sample threshold

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        success = False

    finally:
        # END LOGGING (Mandatory Format: Single Space)
        score = sum(rewards)
        rewards_list = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_count} score={score:.2f} rewards={rewards_list}")
        env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
