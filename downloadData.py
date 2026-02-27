#!/usr/bin/env python3
"""
download_dataset.py
===================
Download datasets from your HuggingFace account to local storage.

Default output directory:
  <capstone-vla>/dataset_tools/data/<dataset_name>/

Usage
-----
  # Interactive — lists your HF datasets and lets you pick
  python download_dataset.py --base /workspace/capstone-vla

  # Direct — skip the picker, download immediately
  python download_dataset.py --base /workspace/capstone-vla --dataset your-username/my-dataset

  # Download all your datasets without prompting
  python download_dataset.py --base /workspace/capstone-vla --all

Optional flags:
  --output  /custom/output/dir   Override default output directory
  --reauth                       Force re-entry of HF token
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed.")
    print("        Run:  pip install huggingface_hub")
    sys.exit(1)

TOKEN_FILENAME = ".hf_token"

# ---------------------------------------------------------------------------
# Token (shared logic with upload_checkpoints.py)
# ---------------------------------------------------------------------------

def resolve_token(base: Path, reauth: bool) -> str:
    import getpass
    token_file = base / TOKEN_FILENAME

    if not reauth:
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            print("[AUTH] Using token from HF_TOKEN environment variable.")
            return env_token

    if not reauth and token_file.exists():
        cached = token_file.read_text().strip()
        if cached:
            print(f"[AUTH] Using cached token from {token_file}")
            return cached

    print()
    print("=" * 60)
    print("  HuggingFace Authentication")
    print("=" * 60)
    print("  Get your token at: https://huggingface.co/settings/tokens")
    print("  Make sure it has  READ  permission (or WRITE if shared with uploader).")
    print()
    token = getpass.getpass("  Paste your HF token (input hidden): ").strip()

    if not token:
        print("[ERROR] No token entered. Exiting.")
        sys.exit(1)

    save = input("  Save token to disk for future runs? [Y/n]: ").strip().lower()
    if save in ("", "y", "yes"):
        token_file.write_text(token)
        token_file.chmod(0o600)
        print(f"  Token saved to {token_file}")
        print(f"  (Run with --reauth to change it)")
    else:
        print("  Token not saved — you will be prompted again next run.")

    print()
    return token


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_base(base_arg: str) -> Path:
    candidates = [
        Path(base_arg),
        Path("/workspace/capstone-vla"),
        Path("/home/sensethreat/lab_mimic/VLA_IL/capstone-vla"),
    ]
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    print("[ERROR] Could not find capstone-vla directory.")
    print(f"        Tried: {[str(c) for c in candidates]}")
    print("        Pass the correct path with --base /path/to/capstone-vla")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset listing + interactive picker
# ---------------------------------------------------------------------------

def list_user_datasets(api: HfApi) -> list:
    """Return all dataset repos owned by the authenticated user."""
    user = api.whoami()
    username = user["name"]
    print(f"[INFO] Fetching datasets for user: {username}")
    datasets = list(api.list_datasets(author=username))
    return datasets


def interactive_picker(datasets: list) -> list:
    """Display numbered list and let user select one or more datasets."""
    if not datasets:
        print("[INFO] No datasets found on your account.")
        sys.exit(0)

    print()
    print("=" * 60)
    print("  Your HuggingFace Datasets")
    print("=" * 60)
    for i, ds in enumerate(datasets, 1):
        size_str = ""
        print(f"  [{i:>2}] {ds.id}  {size_str}")
    print()
    print("  Enter numbers to download (e.g.  1  or  1,3  or  all)")
    raw = input("  Selection: ").strip().lower()

    if raw == "all":
        return datasets

    selected = []
    for part in raw.replace(" ", "").split(","):
        try:
            idx = int(part) - 1
            if 0 <= idx < len(datasets):
                selected.append(datasets[idx])
            else:
                print(f"  [WARN] Index {part} out of range, skipping.")
        except ValueError:
            print(f"  [WARN] Could not parse '{part}', skipping.")

    if not selected:
        print("[ERROR] No valid datasets selected.")
        sys.exit(1)

    return selected


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_dataset(api: HfApi, repo_id: str, output_root: Path, token: str):
    """Download a full dataset repo via snapshot_download."""
    # Output dir: output_root/<dataset_name>  (strip username prefix)
    dataset_name = repo_id.split("/")[-1]
    local_dir = output_root / dataset_name

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Repo   : {repo_id}")
    print(f"  Saving : {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=token,
            ignore_patterns=["*.gitattributes", ".gitattributes"],
        )
        print(f"  [OK] Downloaded to {local_dir}")
    except Exception as e:
        print(f"  [ERROR] Failed to download {repo_id}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets to local storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base",     type=str, default="auto",
                        help="Path to capstone-vla root (auto-detected if omitted)")
    parser.add_argument("--dataset",  type=str, default=None,
                        help="HF dataset repo id to download directly (skips picker)")
    parser.add_argument("--all",      action="store_true",
                        help="Download all datasets on your account without prompting")
    parser.add_argument("--output",   type=str, default=None,
                        help="Override default output directory")
    parser.add_argument("--reauth",   action="store_true",
                        help="Force re-entry of HF token even if one is cached")
    args = parser.parse_args()

    # --- Base path ---
    base = resolve_base(args.base if args.base != "auto" else "")

    # Token file lives in dataset_tools/ alongside this script
    script_dir = Path(__file__).parent
    token = resolve_token(script_dir, args.reauth)

    # --- Output directory ---
    output_root = Path(args.output) if args.output else base / "dataset_tools" / "data"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_root}")

    # --- API client + auth check ---
    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"[AUTH] Logged in as: {user['name']}")
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        print("        Run with --reauth to enter a new token.")
        sys.exit(1)

    # --- Resolve which datasets to download ---
    if args.dataset:
        # CLI mode — skip picker entirely
        targets = [args.dataset]

    elif args.all:
        # Download everything without prompting
        datasets = list_user_datasets(api)
        targets = [ds.id for ds in datasets]
        print(f"[INFO] Downloading all {len(targets)} dataset(s).")

    else:
        # Interactive picker
        datasets = list_user_datasets(api)
        selected = interactive_picker(datasets)
        targets = [ds.id for ds in selected]

    # --- Download ---
    print(f"\n{'='*60}")
    print(f"Downloading {len(targets)} dataset(s)")
    print(f"{'='*60}")

    for repo_id in targets:
        download_dataset(api, repo_id, output_root, token)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for repo_id in targets:
        dataset_name = repo_id.split("/")[-1]
        local_dir = output_root / dataset_name
        if local_dir.exists():
            # Count files and total size
            files = list(local_dir.rglob("*"))
            files = [f for f in files if f.is_file()]
            total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
            print(f"  {repo_id}")
            print(f"    -> {local_dir}  ({len(files)} files, {total_mb:.1f} MB)")
        else:
            print(f"  {repo_id}  -> [FAILED]")


if __name__ == "__main__":
    main()