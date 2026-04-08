---
title: Email Triage RL Environment
emoji: 📧
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# 📧 Email Triage Environment (`email_env`)

> A **multi-step reinforcement learning environment** for AI email triage — built on top of OpenEnv.  
> Agents classify, categorize, and reply to emails across **3 difficulty levels** with shaped rewards.

---

## 🚀 Overview

`email_env` is an OpenEnv-compatible environment where an AI agent must handle real-world email management tasks:

| Task | Difficulty | Actions | Max Reward |
|------|-----------|---------|-----------|
| Spam Detection | 🟢 Easy | `classify` | 1 |
| Category Classification | 🟡 Medium | `classify` + `categorize` | 2 |
| Full Decision Pipeline | 🔴 Hard | `classify` + `categorize` + `reply` | 3 |

---

## 🧠 Key Features

- **Real-world use case** — Email spam detection, topic classification, and professional reply generation
- **Multi-step decision making** — Agent takes sequential actions (up to 3 steps per episode)
- **Reward isolation** — Strict integer reward (`1` or `0`) for each independent step
- **Task difficulty sampling** — Each episode randomly selects `easy`, `medium`, or `hard`
- **Pluggable LLM backend** — Works with any OpenAI-compatible API (HuggingFace, OpenRouter, etc.)

---

## 📂 Project Structure

```
email_env/
├── env.py           # EmailEnv class — multi-step RL environment
├── models.py        # Pydantic models: Action, Observation
├── tasks.py         # Grader functions: grade_spam, grade_category, grade_reply
├── emails.py        # Email dataset (15 diverse emails)
├── inference.py     # LLM agent inference loop
├── openenv.yaml     # OpenEnv configuration
├── Dockerfile       # Docker container definition
├── requirements.txt # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
# Required
export HF_TOKEN=hf_your_token_here

# Optional (defaults shown)
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
```

### 3. Run inference

```bash
python inference.py
```

---

## 🎮 Environment API

### `env.reset()`

Resets the environment. Returns an initial observation and randomly selects task difficulty.

```python
from env import EmailEnv

env = EmailEnv()
result = env.reset()
obs = result["observation"]
print(obs.subject, obs.body)
print(result["info"]["task_type"])  # easy / medium / hard
```

### `env.step(action)`

Takes one action and returns reward, next observation, and done flag.

```python
from models import Action

action = Action(action_type="classify", content="not_spam")
result = env.step(action)

print(result["reward"])             # step reward
print(result["cumulative_reward"])  # total reward so far
print(result["done"])               # True when episode ends
```

---

## 🏆 Task Evaluation

Rewards are **strict integers**, evaluating each action independently:

| Task (`Action`) | Correct | Wrong |
|----------------|---------|-------|
| Spam Detection (`classify`) | +1 | 0 |
| Categorization (`categorize`) | +1 | 0 |
| Professional Reply (`reply`) | +1 | 0 |

**Total max reward = 3 (for hard tasks)** ✅

---

## 🧪 Task Breakdown

### 🟢 Easy — Spam Detection
- Agent receives email, responds with `spam` or `not_spam`
- Clean binary reward

### 🟡 Medium — Spam + Category
- Agent classifies spam and categorizes into `work`, `personal`, or `promotion`
- Strict evaluation explicitly separating categorizations

### 🔴 Hard — Full Pipeline
- Agent classifies, categorizes, AND writes a professional reply
- Parsed from a single structured LLM response

---

## 🐳 Docker

### Build

```bash
docker build -t email_env .
```

### Run

```bash
docker run -e HF_TOKEN=hf_your_token_here email_env
```

---

## ☁️ Deploy to Hugging Face

```bash
huggingface-cli login
openenv push --repo-id your-username/email-env
```

---

## 📊 Sample Output

```
[START] task=email_triage env=openenv model=gpt-4.1-mini
[INFO]  email_id=4 task_type=hard steps=['classify', 'categorize', 'reply']
[STEP]  step=1 action_type=classify content="not_spam" reward=1 done=false error=null
[STEP]  step=2 action_type=categorize content="work" reward=1 done=false error=null
[STEP]  step=3 action_type=reply content="Thank you for the alert. I'll look into it immediately." reward=1 done=true error=null
[END]   success=true steps=3 total_reward=3 rewards=[1,1,1]
```

---

## ✅ Validation Checklist

- [x] 3 tasks implemented (easy / medium / hard)
- [x] Strict boolean integer evaluation points
- [x] Inference logs in correct format
- [x] Multi-step environment (up to 3 steps)
- [x] Docker builds with `--no-cache-dir`
- [x] 15 diverse emails in dataset
- [x] openenv.yaml configured

---

## 📄 License

MIT
