"""
Sanity Test for email_env — no API key required.
Tests: models, graders, all 3 task types, reward shaping, multi-step logic.
"""

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
print("  EMAIL ENV — SANITY TEST")
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
print("\n[2] ENV RESET")
# ─────────────────────────────────────────
env = EmailEnv()
result = env.reset()

obs   = result["observation"]
info  = result["info"]

check("reset returns observation",      obs is not None)
check("observation has subject",        len(obs.subject) > 0)
check("observation has body",           len(obs.body) > 0)
check("observation history is empty",   obs.history == [])
check("task_type in [easy/medium/hard]", info["task_type"] in ["easy", "medium", "hard"])
check("reward is 0.0",                  result["reward"] == 0.0)
check("done is False",                  result["done"] == False)


# ─────────────────────────────────────────
print("\n[3] EASY TASK — classify only")
# ─────────────────────────────────────────
# Force easy mode for determinism
env.task_type = "easy"
env.max_steps = 1
env.step_count = 0

action = Action(action_type="classify", content="spam")
step_result = env.step(action)

check("step returns reward",            "reward" in step_result)
check("reward is float",                isinstance(step_result["reward"], float))
check("reward in [0.0, 0.4]",          0.0 <= step_result["reward"] <= 0.4)
check("done=True after 1 step",         step_result["done"] == True)
check("history has 1 entry",            len(step_result["observation"].history) == 1)


# ─────────────────────────────────────────
print("\n[4] MEDIUM TASK — classify + categorize")
# ─────────────────────────────────────────
env.reset()
env.task_type = "medium"
env.max_steps = 2
env.step_count = 0
env.cumulative_reward = 0.0

r1 = env.step(Action(action_type="classify", content="not_spam"))
check("step 1 not done",               r1["done"] == False)
check("step 1 reward <= 0.4",          r1["reward"] <= 0.4)

r2 = env.step(Action(action_type="categorize", content="work"))
check("step 2 done",                   r2["done"] == True)
check("step 2 reward <= 0.3",          r2["reward"] <= 0.3)
check("cumulative reward <= 0.7",      r2["cumulative_reward"] <= 0.7)


# ─────────────────────────────────────────
print("\n[5] HARD TASK — classify + categorize + reply")
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
check("hard: done after step 3",       h3["done"] == True)
check("hard: cumulative reward <= 1.0", h3["cumulative_reward"] <= 1.0)
check("hard: history length == 3",     len(h3["observation"].history) == 3)


# ─────────────────────────────────────────
print("\n[6] DATASET CHECK")
# ─────────────────────────────────────────
from emails import emails

check("dataset has >= 10 emails",      len(emails) >= 10)
check("all emails have label field",   all("label" in e for e in emails))
check("all emails have category field",all("category" in e for e in emails))
check("labels are valid",              all(e["label"] in ["spam", "not_spam"] for e in emails))
check("categories are valid",          all(e["category"] in ["work", "personal", "promotion"] for e in emails))


# ─────────────────────────────────────────
print("\n" + "="*55)
passed = sum(results)
total  = len(results)
print(f"  RESULT: {passed}/{total} tests passed")
if passed == total:
    print("  🎉 ALL TESTS PASSED — env is ready!")
else:
    print(f"  ⚠️  {total - passed} test(s) failed — review above")
print("="*55 + "\n")
