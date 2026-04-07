"""
deploy.py — Upload email_env to Hugging Face Spaces (Modern upload_folder version)

Usage:
    python deploy.py --token hf_xxx --username your-hf-username
    python deploy.py --token hf_xxx --username your-hf-username --space-name email-triage-env
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Files and directories to ignore during upload
IGNORE_PATTERNS = [
    "__pycache__",
    ".git",
    ".env",
    "deploy.py",
    "*.log",
    "sanity_out.txt",
    "test_api_out.txt",
    "inference_out.txt",
    "test_output.txt",
    "venv",
    ".ipynb_checkpoints"
]

def deploy(token: str, username: str, space_name: str):
    api = HfApi(token=token)
    repo_id = f"{username}/{space_name}"
    project_dir = Path(__file__).parent.absolute()

    print(f"\n[INFO] Deploying to: https://huggingface.co/spaces/{repo_id}")
    print(f"[INFO] Source directory: {project_dir}")

    # 1. Create Space repo (skip if already exists)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",  # Modified to docker as per README/Dockerfile requirement
            token=token,
            exist_ok=True,
            private=False,
        )
        print(f"[OK]   Space created/found: {repo_id}")
    except Exception as e:
        print(f"[ERROR] Could not create Space: {e}")
        return

    # 2. Upload the entire folder
    print(f"\n[INFO] Syncing folder to Hugging Face...")
    try:
        api.upload_folder(
            folder_path=str(project_dir),
            repo_id=repo_id,
            repo_type="space",
            token=token,
            ignore_patterns=IGNORE_PATTERNS,
            delete_patterns=["app.py", "server.py"] # Automatically cleanup obsolete files
        )
        print(f"[OK]   Upload complete!")
    except Exception as e:
        print(f"[FAIL] Deployment failed: {e}")
        return

    print(f"\n{'='*55}")
    print(f"  DEPLOYED SUCCESSFUL!")
    print(f"  URL: https://huggingface.co/spaces/{repo_id}")
    print(f"{'='*55}")
    print("\nNOTE: The Space may take 1-2 minutes to build.")
    print("      Check Logs: https://huggingface.co/spaces/{repo_id}/logs")


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
