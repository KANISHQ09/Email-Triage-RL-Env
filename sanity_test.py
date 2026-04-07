"""
Sanity Test for email_env (OpenEnv Compliant)
Tests: EnvResult attributes, graders, task logic, and multi-step rewards.
"""

import sys
import io

# Fix encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from env import EmailEnv
from models import Action
from tasks import grade_spam, grade_category, grade_reply

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []

def check(label, condition):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    results.append(condition)


print("\n" + "="*55)
print("  EMAIL ENV — OPENENV COMPLIANCE TEST")
print("="*55)


# ─────────────────────────────────────────
print("\n[1] GRADER FUNCTIONS")
# ─────────────────────────────────────────
check("grade_spam correct",        grade_spam("spam", "spam") == 1.0)
check("grade_spam wrong",          grade_spam("not_spam", "spam") == 0.0)
check("grade_spam case-insensitive", grade_spam("SPAM", "spam") == 1.0)

check("grade_category correct",    grade_category("work", "work") == 1.0)
check("grade_category partial",    grade_category("personal", "work") == 0.5)
check("grade_category invalid",    grade_category("gibberish", "work") == 0.0)

check("grade_reply high quality",  grade_reply("Thank you, I'll handle it.") == 1.0)
check("grade_reply noted keyword", grade_reply("Noted, will do.") == 1.0)
check("grade_reply generic",       grade_reply("ok bye") == 0.5)


# ─────────────────────────────────────────
print("\n[2] ENV RESET (EnvResult API)")
# ─────────────────────────────────────────
env = EmailEnv()
result = env.reset()

check("reset returns EnvResult with .observation",  hasattr(result, "observation"))
check("observation has subject",                   len(result.observation.subject) > 0)
check("observation has messages history",          len(result.observation.messages) > 0)
check("task_type in info",                         result.info["task_type"] in ["easy", "medium", "hard"])
check("initial reward is 0.0",                     result.reward == 0.0)
check("done is False",                             result.done == False)


# ─────────────────────────────────────────
print("\n[3] EASY TASK — classify (Step API)")
# ─────────────────────────────────────────
env.task_type = "easy"
env.max_steps = 1
env.step_count = 0

action = Action(action_type="classify", content="spam")
r1 = env.step(action)

check("step returns .reward as float",            isinstance(r1.reward, float))
check("step returns .done indexable",             r1.done == True)
check("history length is 1",                      len(r1.observation.history) == 1)
check("messages length is 3 (sys + user + env)", len(r1.observation.messages) == 3)


# ─────────────────────────────────────────
print("\n[4] HARD TASK — Full Pipeline")
# ─────────────────────────────────────────
env.reset()
env.task_type = "hard"
env.max_steps = 3
env.step_count = 0
env.cumulative_reward = 0.0

h1 = env.step(Action(action_type="classify",   content="spam"))
h2 = env.step(Action(action_type="categorize", content="promotion"))
h3 = env.step(Action(action_type="reply",      content="Thank you for reaching out."))

check("hard: 3 steps completed",       env.step_count == 3)
check("hard: .done after step 3",      h3.done == True)
check("hard: info contains cumulative", h3.info["cumulative_reward"] > 0)
check("hard: cumulative reward <= 1.0", h3.info["cumulative_reward"] <= 1.0)
check("hard: messages count is 7",     len(h3.observation.messages) == 7) # 1 sys + 3*(user+env)


# ─────────────────────────────────────────
print("\n[5] ORACLE STATE API")
# ─────────────────────────────────────────
s = env.state()
check("state() returns dict",           isinstance(s, dict))
check("state() has email data",        "email" in s)
check("state() reflects done status",  s["done"] == True)


# ─────────────────────────────────────────
print("\n" + "="*55)
passed = sum(results)
total  = len(results)
print(f"  RESULT: {passed}/{total} tests passed")
if passed == total:
    print("  🎉 OPENENV COMPLIANCE VERIFIED!")
else:
    print(f"  ⚠️  {total - passed} test(s) failed — review above")
print("="*55 + "\n")
