import os
import re
import sys
from openai import OpenAI
from env import EmailEnv
from models import Action

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # Optional

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required. Set it before running.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# --- LLM Call ---
def get_llm_response(prompt: str) -> str:
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()


# --- Prompt Templates ---
def build_classify_prompt(obs) -> str:
    return f"""You are an email assistant. Classify the following email as spam or not_spam.

Email:
Subject: {obs.subject}
Body: {obs.body}
Sender: {obs.sender}

Respond with ONLY one word: spam or not_spam"""


def build_categorize_prompt(obs) -> str:
    return f"""You are an email assistant. Categorize the following email into one of: work, personal, promotion.

Email:
Subject: {obs.subject}
Body: {obs.body}
Sender: {obs.sender}

Respond with ONLY one word: work, personal, or promotion"""


def build_reply_prompt(obs) -> str:
    return f"""You are a professional email assistant. Write a brief, professional reply to this email.

Email:
Subject: {obs.subject}
Body: {obs.body}
Sender: {obs.sender}

Write a concise reply (1-3 sentences). Be professional and acknowledge the email."""


def build_full_prompt(obs) -> str:
    return f"""You are an intelligent email triage assistant. Analyze this email and provide:
1. Classification: is it spam or not_spam?
2. Category: work, personal, or promotion?
3. A brief professional reply.

Email:
Subject: {obs.subject}
Body: {obs.body}
Sender: {obs.sender}
History: {obs.history}

Respond in EXACTLY this format:
classify: <spam or not_spam>
categorize: <work or personal or promotion>
reply: <your reply here>"""


def parse_full_response(text: str) -> dict:
    """Parse multi-line LLM response into action components."""
    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("classify:"):
            result["classify"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("categorize:"):
            result["categorize"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reply:"):
            result["reply"] = line.split(":", 1)[1].strip()
    return result


# --- Main Inference Loop ---
def main():
    env = EmailEnv()

    print(f"[START] task=email_triage env=openenv model={MODEL_NAME}")

    result = env.reset()
    obs = result["observation"]
    task_type = result["info"]["task_type"]
    expected_steps = result["info"]["expected_steps"]

    print(f"[INFO]  email_id={obs.email_id} task_type={task_type} steps={expected_steps}")

    rewards = []
    step = 0

    # ---- HARD: full multi-action pipeline with single prompt ----
    if task_type == "hard":
        full_prompt = build_full_prompt(obs)
        raw_response = get_llm_response(full_prompt)
        parsed = parse_full_response(raw_response)

        action_sequence = [
            ("classify", parsed.get("classify", "not_spam")),
            ("categorize", parsed.get("categorize", "work")),
            ("reply", parsed.get("reply", "Thank you for your email.")),
        ]

        for action_type, content in action_sequence:
            step += 1
            action = Action(action_type=action_type, content=content)
            result = env.step(action)
            reward = round(result["reward"], 2)
            done = result["done"]
            rewards.append(reward)
            print(f"[STEP]  step={step} action_type={action_type} content=\"{content[:60]}\" reward={reward:.2f} done={str(done).lower()} error=null")
            if done:
                break

    # ---- MEDIUM: classify + categorize ----
    elif task_type == "medium":
        classify_response = get_llm_response(build_classify_prompt(obs))
        step += 1
        action = Action(action_type="classify", content=classify_response)
        result = env.step(action)
        reward = round(result["reward"], 2)
        rewards.append(reward)
        print(f"[STEP]  step={step} action_type=classify content=\"{classify_response}\" reward={reward:.2f} done={str(result['done']).lower()} error=null")

        if not result["done"]:
            obs = result["observation"]
            categorize_response = get_llm_response(build_categorize_prompt(obs))
            step += 1
            action = Action(action_type="categorize", content=categorize_response)
            result = env.step(action)
            reward = round(result["reward"], 2)
            rewards.append(reward)
            print(f"[STEP]  step={step} action_type=categorize content=\"{categorize_response}\" reward={reward:.2f} done={str(result['done']).lower()} error=null")

    # ---- EASY: spam classification only ----
    else:
        classify_response = get_llm_response(build_classify_prompt(obs))
        step += 1
        action = Action(action_type="classify", content=classify_response)
        result = env.step(action)
        reward = round(result["reward"], 2)
        rewards.append(reward)
        print(f"[STEP]  step={step} action_type=classify content=\"{classify_response}\" reward={reward:.2f} done={str(result['done']).lower()} error=null")

    # ---- Summary ----
    total_reward = round(sum(rewards), 2)
    success = total_reward > 0.5
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(f"[END]   success={str(success).lower()} steps={len(rewards)} total_reward={total_reward:.2f} rewards=[{rewards_str}]")


if __name__ == "__main__":
    main()
