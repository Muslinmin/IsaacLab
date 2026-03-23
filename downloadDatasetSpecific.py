#!/usr/bin/env python3
"""
download_dataset.py
===================
Download datasets (or specific files/folders) from HuggingFace.

Token resolution: HF_TOKEN env var → /workspace/IsaacLab/.hf_token → interactive prompt

Usage
-----
  # Download entire dataset repo
  python download_dataset.py --dataset Lusmse/sd

  # Download a specific file
  python download_dataset.py --dataset Lusmse/sd --file path/to/file.hdf5

  # Download multiple specific files
  python download_dataset.py --dataset Lusmse/sd --file demo_0.hdf5 --file demo_1.hdf5 --file demo_2.hdf5

  # Download a specific folder
  python download_dataset.py --dataset Lusmse/sd --folder some_subfolder

  # Download by glob pattern (e.g. all HDF5 files)
  python download_dataset.py --dataset Lusmse/sd --pattern "*.hdf5"

  # Multiple patterns
  python download_dataset.py --dataset Lusmse/sd --pattern "*.hdf5" --pattern "*.json"

  # Override output directory
  python download_dataset.py --dataset Lusmse/sd --output /custom/path

  # Interactive mode — lists your datasets and lets you pick
  python download_dataset.py

  # Force re-authentication
  python download_dataset.py --dataset Lusmse/sd --reauth
"""

import argparse
import os
import sys
import getpass
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed.")
    print("        Run:  pip install huggingface_hub")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
TOKEN_FILE = SCRIPT_DIR / ".hf_token"
DEFAULT_OUTPUT = SCRIPT_DIR / "datasets"


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------
def resolve_token(reauth: bool = False) -> str:
    if not reauth:
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            print("[AUTH] Using token from HF_TOKEN env var.")
            return env_token

    if not reauth and TOKEN_FILE.exists():
        cached = TOKEN_FILE.read_text().strip()
        if cached:
            print(f"[AUTH] Using cached token from {TOKEN_FILE}")
            return cached

    print()
    print("=" * 60)
    print("  HuggingFace Authentication")
    print("=" * 60)
    print("  Get your token at: https://huggingface.co/settings/tokens")
    print()
    token = getpass.getpass("  Paste your HF token (input hidden): ").strip()
    if not token:
        print("[ERROR] No token entered. Exiting.")
        sys.exit(1)

    save = input("  Save token to disk? [Y/n]: ").strip().lower()
    if save in ("", "y", "yes"):
        TOKEN_FILE.write_text(token)
        TOKEN_FILE.chmod(0o600)
        print(f"  Token saved to {TOKEN_FILE}")
    print()
    return token


# ---------------------------------------------------------------------------
# Interactive picker
# ---------------------------------------------------------------------------
def interactive_picker(api: HfApi) -> str:
    user = api.whoami()
    username = user["name"]
    print(f"[INFO] Fetching datasets for user: {username}")
    datasets = list(api.list_datasets(author=username))

    if not datasets:
        print("[INFO] No datasets found on your account.")
        sys.exit(0)

    print()
    print("=" * 60)
    print("  Your HuggingFace Datasets")
    print("=" * 60)
    for i, ds in enumerate(datasets, 1):
        print(f"  [{i:>2}] {ds.id}")
    print()
    raw = input("  Select a dataset number: ").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(datasets):
            return datasets[idx].id
    except ValueError:
        pass
    print("[ERROR] Invalid selection.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# List remote files (for --ls)
# ---------------------------------------------------------------------------
def list_remote_files(api: HfApi, repo_id: str, token: str, folder: str = None):
    """List all files in a dataset repo (optionally filtered to a subfolder)."""
    print(f"\n[INFO] Files in {repo_id}:")
    print("-" * 60)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    for f in sorted(files):
        if folder and not f.startswith(folder.rstrip("/") + "/"):
            continue
        print(f"  {f}")
    print(f"-" * 60)
    print(f"  Total: {len(files)} file(s)")


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------
def do_download(repo_id: str, token: str, output_dir: Path,
                files: list = None, folder: str = None, patterns: list = None):
    dataset_name = repo_id.split("/")[-1]
    local_dir = output_dir / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Repo   : {repo_id}")
    print(f"  Output : {local_dir}")

    # --- Specific file(s) download ---
    if files:
        print(f"  Files  : {len(files)} file(s)")
        print("=" * 60)
        ok, fail = 0, 0
        for f in files:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f,
                    local_dir=str(local_dir),
                    token=token,
                )
                print(f"  [OK]    {f}")
                ok += 1
            except Exception as e:
                print(f"  [ERROR] {f} — {e}")
                fail += 1
        print(f"\n  Done: {ok} downloaded, {fail} failed")
        return

    # --- Snapshot download (full repo, folder, or pattern) ---
    allow = None
    if folder:
        allow = [f"{folder.rstrip('/')}/*"]
        print(f"  Folder : {folder}")
    elif patterns:
        allow = list(patterns)
        print(f"  Pattern: {allow}")
    else:
        print(f"  Scope  : entire repo")

    print("=" * 60)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=allow,
            token=token,
            ignore_patterns=["*.gitattributes", ".gitattributes"],
        )
        # Summary
        files = [f for f in local_dir.rglob("*") if f.is_file()]
        total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
        print(f"  [OK] {len(files)} files, {total_mb:.1f} MB total")
    except Exception as e:
        print(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets (full or partial).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="HF repo id (e.g. Lusmse/sd). Omit for interactive picker.")
    parser.add_argument("--file", type=str, action="append", default=None,
                        help="Download specific file(s) by repo-relative path (repeatable).")
    parser.add_argument("--folder", type=str, default=None,
                        help="Download only files under this subfolder.")
    parser.add_argument("--pattern", type=str, action="append", default=None,
                        help="Glob pattern to filter files (repeatable). e.g. '*.hdf5'")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output root directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--ls", action="store_true",
                        help="List remote files in the dataset instead of downloading.")
    parser.add_argument("--reauth", action="store_true",
                        help="Force re-entry of HF token.")

    args = parser.parse_args()
    token = resolve_token(args.reauth)
    api = HfApi(token=token)

    # Verify auth
    try:
        user = api.whoami()
        print(f"[AUTH] Logged in as: {user['name']}")
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        print("        Run with --reauth to re-authenticate.")
        sys.exit(1)

    # Resolve dataset
    repo_id = args.dataset or interactive_picker(api)

    # List-only mode
    if args.ls:
        list_remote_files(api, repo_id, token, folder=args.folder)
        return

    # Download
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT
    do_download(
        repo_id=repo_id,
        token=token,
        output_dir=output_dir,
        files=args.file,
        folder=args.folder,
        patterns=args.pattern,
    )


if __name__ == "__main__":
    main()