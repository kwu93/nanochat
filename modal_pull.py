"""
Sync nanochat artifacts (tokenizer, model checkpoints, report) from the Modal
volume down to this machine, so you can inspect and chat with checkpoints
locally while a training run is still going on Modal.

Checkpoints are immutable once written (filenames contain the step), so each is
downloaded at most once. Optimizer shards (optim_*.pt) are skipped: they are
only needed to resume training on Modal, not for inference. A checkpoint is
only pulled once its meta_*.json is visible in the volume: save_checkpoint()
writes the model file before the meta file, and Modal volume background commits
are point-in-time snapshots, so a visible meta file implies a complete model file.

Requires the `modal` CLI (same auth as modal_speedrun.py), stdlib only otherwise.

Usage:

    python modal_pull.py                 # one-shot sync to ~/.cache/nanochat-modal
    python modal_pull.py --watch 300     # keep syncing every 5 minutes
    python modal_pull.py --dest DIR      # sync to a different directory

Then chat with a pulled checkpoint from the repo root, e.g.:

    NANOCHAT_BASE_DIR=~/.cache/nanochat-modal python -m scripts.chat_web \
        -i base --model-tag d24 --step 2000
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

VOLUME = "nanochat-cache"
DEFAULT_DEST = Path.home() / ".cache" / "nanochat-modal"

# Small, mutable files: re-downloaded on every sync (a few hundred KiB total).
SMALL_DIRS = ["tokenizer", "report"]
# Checkpoint directories, keyed by the same names checkpoint_manager.py uses.
CHECKPOINT_DIRS = ["base_checkpoints", "chatsft_checkpoints", "chatrl_checkpoints"]

META_RE = re.compile(r"meta_(\d{6})\.json$")


def volume_ls(path: str):
    """List a volume directory; returns None if the path doesn't exist (yet)."""
    proc = subprocess.run(
        ["modal", "volume", "ls", VOLUME, path, "--json"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def volume_get(remote: str, local_dir: Path) -> None:
    """Download one file from the volume into local_dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", VOLUME, remote, f"{local_dir}/", "--force"],
        check=True, capture_output=True, text=True,
    )


def sync(dest: Path) -> list[str]:
    """One sync pass; returns the volume paths of newly pulled checkpoint files."""
    for dirname in SMALL_DIRS:
        for entry in volume_ls(dirname) or []:
            if entry["type"] == "file":
                volume_get(entry["filename"], dest / dirname)

    new_files = []
    for group in CHECKPOINT_DIRS:
        for tag_entry in volume_ls(group) or []:
            if tag_entry["type"] != "dir":
                continue
            tag_path = tag_entry["filename"]  # e.g. "base_checkpoints/d24"
            local_tag_dir = dest / tag_path
            remote_names = {
                Path(e["filename"]).name
                for e in volume_ls(tag_path) or []
                if e["type"] == "file"
            }
            # Only steps whose meta file is visible are guaranteed complete.
            steps = sorted(m.group(1) for n in remote_names if (m := META_RE.match(n)))
            for step in steps:
                for name in (f"model_{step}.pt", f"meta_{step}.json"):
                    if name in remote_names and not (local_tag_dir / name).exists():
                        print(f"  pulling {tag_path}/{name} ...", flush=True)
                        volume_get(f"{tag_path}/{name}", local_tag_dir)
                        new_files.append(f"{tag_path}/{name}")
    return new_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"local NANOCHAT_BASE_DIR to sync into (default: {DEFAULT_DEST})")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="keep syncing every SECONDS (0 = sync once and exit)")
    args = parser.parse_args()

    while True:
        print(f"Syncing volume '{VOLUME}' -> {args.dest}")
        new_files = sync(args.dest)
        if new_files:
            print(f"Pulled {len(new_files)} new file(s). Serve the latest with e.g.:")
            print(f"  NANOCHAT_BASE_DIR={args.dest} python -m scripts.chat_web -i base|sft "
                  f"--model-tag <tag> --step <step>")
        else:
            print("Already up to date, no new checkpoints.")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
