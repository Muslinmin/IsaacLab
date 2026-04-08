#!/usr/bin/env python3
"""
upload_sd.py
============
Upload a file or folder to the HuggingFace dataset repo: Lusmse/sd

Files are placed inside a dated folder on HF:
  <YYYYMMDD_HHMM>_<first4chars>/<filename_or_foldername>/

Examples
--------
  # Upload a folder (auto-detects default source dirs)
  python upload_sd.py --path /workspace/IsaacLab/dataset_tools/data/sd/my_recording

  # Upload a single file
  python upload_sd.py --path /workspace/IsaacLab/dataset_tools/data/sd/episode_001.hdf5

  # Override the HF repo (default: Lusmse/sd)
  python upload_sd.py --path /some/folder --repo other-username/other-repo

  # Force re-entry of HF token
  python upload_sd.py --path /some/folder --reauth

Token
-----
  Looks for .hf_token in this order:
    1. HF_TOKEN environment variable
    2. .hf_token file next to this script
    3. .hf_token file in the detected default data dir
    4. Interactive prompt (optionally saved to disk)
"""

import argparse
import getpass
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("[ERROR] huggingface_hub is not installed.")
    print("        Run:  pip install huggingface_hub")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_REPO      = "Lusmse/sd"
TOKEN_FILENAME    = ".hf_token"

# Default source/data directories (checked in order for token file fallback)
DEFAULT_DATA_DIRS = [
    Path("/workspace/IsaacLab/dataset_tools/data/sd"),
    Path("/home/sensethreat/lab_mimic/IsaacLab/datasets"),
]

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def find_token_file() -> Path | None:
    """Search for .hf_token next to this script, then in default data dirs."""
    candidates = [
        Path(__file__).parent / TOKEN_FILENAME,
        *[d / TOKEN_FILENAME for d in DEFAULT_DATA_DIRS],
        *[d.parent / TOKEN_FILENAME for d in DEFAULT_DATA_DIRS],
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_token(reauth: bool) -> tuple[str, Path | None]:
    """
    Returns (token, token_file_path_if_found).
    Resolution order:
      1. HF_TOKEN env var
      2. .hf_token file (next to script or in data dirs)
      3. Interactive prompt
    """
    if not reauth:
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            print("[AUTH] Using token from HF_TOKEN environment variable.")
            return env_token, None

    if not reauth:
        token_file = find_token_file()
        if token_file:
            cached = token_file.read_text().strip()
            if cached:
                print(f"[AUTH] Using cached token from {token_file}")
                return cached, token_file

    # Interactive prompt
    print()
    print("=" * 60)
    print("  HuggingFace Authentication")
    print("=" * 60)
    print("  Get your token at: https://huggingface.co/settings/tokens")
    print("  Make sure it has  WRITE  permission.")
    print()
    token = getpass.getpass("  Paste your HF token (input hidden): ").strip()

    if not token:
        print("[ERROR] No token entered. Exiting.")
        sys.exit(1)

    # Offer to save next to this script
    default_save_path = Path(__file__).parent / TOKEN_FILENAME
    save = input(f"  Save token to {default_save_path}? [Y/n]: ").strip().lower()
    if save in ("", "y", "yes"):
        default_save_path.write_text(token)
        default_save_path.chmod(0o600)
        print(f"  Token saved to {default_save_path}")
        print(f"  (Run with --reauth to change it)")
        return token, default_save_path
    else:
        print("  Token not saved — you will be prompted again next run.")

    print()
    return token, None


# ---------------------------------------------------------------------------
# HF path helpers
# ---------------------------------------------------------------------------

def make_hf_prefix(target: Path) -> str:
    """
    Build the dated folder name for the HF repo:
      YYYYMMDD_HHMM_<first4chars of target name>

    e.g. target = 'realWorldPouring_v2'  ->  '20250228_1430_real'
    """
    now = datetime.now().strftime("%Y%m%d_%H%M")
    name_slug = target.name[:4].lower()
    return f"{now}_{name_slug}"


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------

def upload_path(api: HfApi, repo_id: str, target: Path, token: str, dry_run: bool):
    """Upload a file or folder to the HF dataset repo under a dated prefix."""

    hf_prefix = make_hf_prefix(target)

    if target.is_file():
        files = [target]
        # Single file: date_prefix/filename
        def repo_path(f: Path) -> str:
            return f"{hf_prefix}/{f.name}"

    elif target.is_dir():
        files = sorted([f for f in target.rglob("*") if f.is_file()])
        # Folder: date_prefix/foldername/relative_path
        def repo_path(f: Path) -> str:
            rel = f.relative_to(target.parent)
            return f"{hf_prefix}/{rel}".replace("\\", "/")
    else:
        print(f"[ERROR] Path does not exist: {target}")
        sys.exit(1)

    if not files:
        print(f"[WARN] No files found in {target}")
        return

    print(f"\n  Source  : {target}")
    print(f"  HF path : {repo_id}/{hf_prefix}/")
    print(f"  Files   : {len(files)}")

    for local_file in files:
        path_in_repo = repo_path(local_file)
        size_mb = local_file.stat().st_size / (1024 ** 2)
        tag = "[DRY-RUN] " if dry_run else ""
        print(f"  {tag}-> {path_in_repo}  ({size_mb:.1f} MB)")

        if not dry_run:
            try:
                api.upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {hf_prefix}",
                )
            except Exception as e:
                print(f"  [ERROR] Failed to upload {path_in_repo}: {e}")

    if not dry_run:
        print(f"\n  [OK] Upload complete.")
        print(f"  View: https://huggingface.co/datasets/{repo_id}/tree/main/{hf_prefix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload a file or folder to Lusmse/sd on HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--path",    type=str, required=True,
                        help="Local file or folder to upload")
    parser.add_argument("--repo",    type=str, default=DEFAULT_REPO,
                        help=f"HF dataset repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without uploading")
    parser.add_argument("--reauth",  action="store_true",
                        help="Force re-entry of HF token even if one is cached")
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"[ERROR] Path does not exist: {target}")
        sys.exit(1)

    # --- Token ---
    token, _ = resolve_token(args.reauth)

    # --- API + auth check ---
    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"[AUTH] Logged in as: {user['name']}")
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        print("        Run with --reauth to enter a new token.")
        sys.exit(1)

    # --- Ensure repo exists ---
    if not args.dry_run:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            private=False,
            exist_ok=True,
            token=token,
        )
        print(f"[INFO] Repo ready : https://huggingface.co/datasets/{args.repo}")
    else:
        print(f"[DRY-RUN] Would verify/create repo: {args.repo}")

    # --- Upload ---
    print(f"\n{'='*60}")
    print(f"Uploading: {target.name}")
    print(f"Repo     : {args.repo}")
    print(f"Dry run  : {args.dry_run}")
    print(f"{'='*60}")

    upload_path(api, args.repo, target, token, args.dry_run)


if __name__ == "__main__":
    main()