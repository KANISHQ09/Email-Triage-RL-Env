from openai import OpenAI

TOKEN = "your hf-token"

# HF router is the new unified endpoint
ROUTER_BASE = "https://router.huggingface.co/hf-inference/v1"

# Models available on HF free serverless tier (2025)
models_to_try = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

print(f"Testing HF Router: {ROUTER_BASE}\n")
working = None
client = OpenAI(base_url=ROUTER_BASE, api_key=TOKEN)

for model in models_to_try:
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say the single word: hello"}],
            max_tokens=15
        )
        reply = res.choices[0].message.content.strip()
        print(f"[OK]   {model}")
        print(f"       Response: {reply}")
        working = model
        break
    except Exception as e:
        err = str(e)[:120]
        print(f"[FAIL] {model}")
        print(f"       {err}")

print()
if working:
    print("=" * 50)
    print("USE THESE SETTINGS IN inference.py:")
    print(f"  API_BASE_URL = {ROUTER_BASE}")
    print(f"  MODEL_NAME   = {working}")
    print("=" * 50)
else:
    print("No working model found.")
    print("Options:")
    print("  1. Your token may need 'Make calls to the serverless Inference API' permission")
    print("     -> Go to: https://huggingface.co/settings/tokens")
    print("     -> Create a Fine-grained token with Inference API enabled")
    print("  2. Or use a different provider (OpenRouter, Together AI, etc.)")
