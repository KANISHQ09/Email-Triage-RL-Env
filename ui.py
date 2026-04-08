import os
import sys
import gradio as gr
from env import EmailEnv
from models import Action
from tasks import grade_spam, grade_category, grade_reply

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# ── Global env instance ────────────────────────────────────────────
env = EmailEnv()
current_result = None

TASK_LABELS = {
    "easy":   "🟢 Easy   — Spam Detection",
    "medium": "🟡 Medium — Spam + Category",
    "hard":   "🔴 Hard   — Full Pipeline (Classify + Categorize + Reply)",
}


# ── Helper: Try LLM call if token present ─────────────────────────
def try_llm(prompt: str) -> str | None:
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=token
        )
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error: {e}]"


# ── Step 1: Load a new email ───────────────────────────────────────
def load_email(task_mode: str):
    global current_result, env

    env = EmailEnv()
    if task_mode != "random":
        env.task_type = task_mode
    
    r = env.reset()
    obs = r.observation
    task = env.task_type
    current_result = r

    email_display = f"""**Subject:** {obs.subject}
**From:** {obs.sender}
**Body:** {obs.body}"""

    expected = " → ".join(env.TASK_STEPS[task])
    task_display = f"{TASK_LABELS[task]}\n\nExpected actions: `{expected}`"
    steps_display = "No steps taken yet."

    return (
        email_display,
        task_display,
        steps_display,
        "",    # classify input
        "",    # categorize input
        "",    # reply input
        "**Total Reward:** 0",
    )


# ── Step 2: Run one or all actions ────────────────────────────────
def run_actions(classify_val: str, categorize_val: str, reply_val: str):
    global current_result, env

    if current_result is None:
        return "⚠️ Load an email first.", "**Total Reward:** 0.00"

    task = env.task_type
    steps_needed = env.TASK_STEPS[task]
    log_lines = []

    inputs = {
        "classify":   classify_val.strip(),
        "categorize": categorize_val.strip(),
        "reply":      reply_val.strip(),
    }

    # If LLM is available and fields are empty, use LLM
    obs = current_result.observation
    for action_type in steps_needed:
        if not inputs[action_type]:
            if action_type == "classify":
                prompt = f"Classify this email as spam or not_spam.\nSubject: {obs.subject}\nBody: {obs.body}\nReply with ONLY: spam or not_spam"
            elif action_type == "categorize":
                prompt = f"Categorize this email as work, personal, or promotion.\nSubject: {obs.subject}\nBody: {obs.body}\nReply with ONLY one word."
            else:
                prompt = f"Write a brief professional reply to:\nSubject: {obs.subject}\nBody: {obs.body}"
            llm_resp = try_llm(prompt)
            if llm_resp:
                inputs[action_type] = llm_resp

    for action_type in steps_needed:
        content = inputs[action_type]
        if not content:
            log_lines.append(f"⚠️ **{action_type}**: No value provided.")
            continue

        action = Action(action_type=action_type, content=content)
        result = env.step(action)
        reward = result.reward
        cumulative = result.info.get("cumulative_reward", 0.0)
        done = result.done

        status = "✅" if reward > 0 else "❌"
        log_lines.append(
            f"{status} **Step {env.step_count}** | `{action_type}` → `{content[:60]}`\n"
            f"   Reward: **{reward:.2f}** | Cumulative: **{cumulative:.2f}** | Done: `{done}`"
        )

    steps_md = "\n\n".join(log_lines) if log_lines else "No valid actions ran."
    total = env.cumulative_reward
    reward_color = "🟢" if total >= 1.5 else ("🟡" if total >= 0.5 else "🔴")
    reward_md = f"**{reward_color} Total Reward: {total:.2f} / {len(steps_needed)}**"

    return steps_md, reward_md


# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(
    title="📧 Email Triage RL Environment",
) as demo:

    gr.Markdown(
        """<div id='header'>
        <h1>📧 Email Triage RL Environment</h1>
        <p>A multi-step reinforcement learning environment for AI email triage.<br>
        Classify, categorize, and reply to emails across 3 difficulty levels.</p>
        </div>"""
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["random", "easy", "medium", "hard"],
            value="random",
            label="Task Difficulty",
            info="Random picks a difficulty each episode"
        )
        load_btn = gr.Button("📨 Load New Email", variant="primary", scale=2)

    with gr.Row():
        with gr.Column(scale=1):
            email_box = gr.Markdown("*Click 'Load New Email' to start.*", label="📩 Email")
            task_box  = gr.Markdown("", label="🎯 Task Info")

        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Actions")
            classify_in   = gr.Textbox(label="classify   (spam / not_spam)",        placeholder="e.g. not_spam")
            categorize_in = gr.Textbox(label="categorize (work / personal / promotion)", placeholder="e.g. work")
            reply_in      = gr.Textbox(label="reply      (professional reply)",      placeholder="e.g. Thank you, noted.", lines=3)

            gr.Markdown(
                "> 💡 **Leave blank** to use LLM auto-fill (requires `HF_TOKEN` env var).  \n"
                "> Or type your own answers and click Run."
            )
            run_btn = gr.Button("▶️ Run Actions", variant="primary")

    steps_box  = gr.Markdown("", label="📊 Step Log")
    reward_box = gr.Markdown("**Total Reward:** 0", elem_classes=["reward-box"])

    gr.Markdown("""---
### 🏆 Task Evaluation
| Task (`Action`) | Correct | Wrong |
|---|---|---|
| Spam Detection (`classify`) | +1 | 0 |
| Categorization (`categorize`) | +1 | 0 |
| Professional Reply (`reply`) | +1 | 0 |

**Max reward = 3 (for hard tasks)** &nbsp;|&nbsp; Built with [OpenEnv](https://github.com/openenv)
""")

    # Wire up events
    load_btn.click(
        fn=load_email,
        inputs=[task_dropdown],
        outputs=[email_box, task_box, steps_box, classify_in, categorize_in, reply_in, reward_box]
    )

    run_btn.click(
        fn=run_actions,
        inputs=[classify_in, categorize_in, reply_in],
        outputs=[steps_box, reward_box]
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css="""
        #header { text-align: center; margin-bottom: 10px; }
        .reward-box { font-size: 1.3em; font-weight: bold; text-align: center; padding: 12px; }
        """
    )
