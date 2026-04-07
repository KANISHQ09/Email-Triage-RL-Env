from huggingface_hub import HfApi
import sys

def cleanup(token, repo_id):
    api = HfApi(token=token)
    files_to_delete = ["app.py", "server.py"]
    
    for f in files_to_delete:
        try:
            print(f"[INFO] Deleting '{f}' from {repo_id}...")
            api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="space")
        except Exception as e:
            print(f"[WARN] Error deleting '{f}': {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cleanup_hf.py <token> <repo_id>")
    else:
        cleanup(sys.argv[1], sys.argv[2])
