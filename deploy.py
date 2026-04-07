"""
deploy.py — Upload email_env to Hugging Face Spaces

Usage:
    python deploy.py --token hf_xxx --username your-hf-username
    python deploy.py --token hf_xxx --username your-hf-username --space-name email-env
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Files to exclude from upload
EXCLUDE = {
    "__pycache__", ".git", ".env",
    "test_api.py", "sanity_test.py", "deploy.py",
    ".gitignore"
}

def deploy(token: str, username: str, space_name: str):
    api = HfApi(token=token)

    repo_id = f"{username}/{space_name}"
    print(f"\n[INFO] Deploying to: https://huggingface.co/spaces/{repo_id}")

    # 1. Create Space repo (skip if already exists)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True,
            private=False,
        )
        print(f"[OK]   Space created/found: {repo_id}")
    except Exception as e:
        print(f"[ERROR] Could not create Space: {e}")
        return

    # 2. Collect files to upload
    project_dir = Path(__file__).parent
    files_to_upload = []

    for f in project_dir.iterdir():
        if f.name in EXCLUDE or f.name.startswith("."):
            continue
        if f.is_file():
            files_to_upload.append(f)

    print(f"\n[INFO] Files to upload ({len(files_to_upload)}):")
    for f in files_to_upload:
        print(f"       {f.name}  ({f.stat().st_size} bytes)")

    # 3. Upload each file
    print("\n[INFO] Uploading...")
    for f in files_to_upload:
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
            print(f"[OK]   Uploaded: {f.name}")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")

    print(f"\n{'='*55}")
    print(f"  DEPLOYED!")
    print(f"  URL: https://huggingface.co/spaces/{repo_id}")
    print(f"{'='*55}")
    print("\nNOTE: The Space may take 1-2 minutes to build.")
    print("      Add HF_TOKEN as a Secret in Space Settings")
    print("      to enable LLM auto-fill feature.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy email_env to HF Spaces")
    parser.add_argument("--token",      required=True,              help="HuggingFace WRITE token (hf_xxx)")
    parser.add_argument("--username",   required=True,              help="Your HuggingFace username")
    parser.add_argument("--space-name", default="email-triage-env", help="Space name (default: email-triage-env)")
    args = parser.parse_args()

    deploy(
        token=args.token,
        username=args.username,
        space_name=args.space_name,
    )
