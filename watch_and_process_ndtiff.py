#!/usr/bin/env python3
"""
watch_and_process_ndtiff.py — simple polling watcher for NDTiff datasets (recursive).

Debug-friendly behavior:
- Processes only datasets that DO NOT yet have auto_process.log in them.
  → Delete auto_process.log to force a re-run.

Assumptions:
- ndstorage is installed; completeness check uses Dataset.is_finished() only.
- This script lives in the SAME folder as stitch_ndtiff.py and dzi_from_bigtiff.py.
- After a dataset is finished, it runs:
    1) python stitch_ndtiff.py <dataset_folder> --scale <SCALE> --channel <CHANNEL>
    2) python dzi_from_bigtiff.py <dataset_folder>\\stitched --split-channel
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable
from ndstorage import Dataset

# ------------------ Fixed config (edit if desired) ------------------
POLL_SECONDS   = 60                      # check once per minute
GRACE_AFTER_FINISH_SECONDS = 60          # wait after is_finished() before processing
INDEX_NAME     = "NDTiff.index"          # identifies a dataset root
LOG_NAME       = "auto_process.log"      # per-dataset log (presence gates processing)
WRITE_MASTER_LOG = True                  # also keep one global log in BASE_DIR
# --------------------------------------------------------------------

# Will be filled from CLI:
BASE_DIR: Path
SCALE: str
CHANNEL: str
MASTER_LOG_PATH: Path

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_master(msg: str) -> None:
    line = f"[{ts()}] {msg}\n"
    print(line, end="")
    if WRITE_MASTER_LOG:
        try:
            MASTER_LOG_PATH.open("a", encoding="utf-8").write(line)
        except Exception:
            pass

def log_ds(ds_dir: Path, msg: str) -> None:
    """Write to dataset log + master log."""
    line = f"[{ts()}][{ds_dir.name}] {msg}\n"
    print(line, end="")
    try:
        (ds_dir / LOG_NAME).open("a", encoding="utf-8").write(line)
    except Exception:
        pass
    if WRITE_MASTER_LOG:
        try:
            MASTER_LOG_PATH.open("a", encoding="utf-8").write(line)
        except Exception:
            pass

def list_candidate_datasets_recursive(base: Path) -> Iterable[Path]:
    """
    Recursively scan for folders that contain NDTiff.index and do NOT yet have auto_process.log.
    Uses os.walk to avoid missing nested datasets; returns dataset folder paths.
    """
    if not base.exists():
        return []
    for root, dirs, files in os.walk(base):
        # Optional: skip hidden/system dirs (uncomment if useful)
        # dirs[:] = [d for d in dirs if not d.startswith('.')]
        if INDEX_NAME in files and LOG_NAME not in files:
            yield Path(root)

def run_processing(ds_dir: Path, script_dir: Path) -> bool:
    """Run the two steps; return True on success."""
    stitch_script = script_dir / "stitch_ndtiff.py"
    dzi_script    = script_dir / "dzi_from_bigtiff.py"

    # Step 1: stitch
    cmd1 = [sys.executable, str(stitch_script), str(ds_dir), "--scale", SCALE, "--channel", CHANNEL]
    log_ds(ds_dir, f"Running: {' '.join(cmd1)}")
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError as e:
        log_ds(ds_dir, f"ERROR: stitch_ndtiff failed with code {e.returncode}")
        return False

    # Step 2: DZI from stitched
    stitched_dir = ds_dir / "stitched"
    cmd2 = [sys.executable, str(dzi_script), str(stitched_dir), "--split-channel"]
    log_ds(ds_dir, f"Running: {' '.join(cmd2)}")
    try:
        subprocess.run(cmd2, check=True)
    except subprocess.CalledProcessError as e:
        log_ds(ds_dir, f"ERROR: dzi_from_bigtiff failed with code {e.returncode}")
        return False

    return True

def parse_args():
    ap = argparse.ArgumentParser(description="Watch for finished NDTiff datasets (recursively) and process them.")
    ap.add_argument("--base-dir", required=True, help="Root folder to scan recursively for NDTiff datasets")
    ap.add_argument("--scale", default="1.0", help="Scale passed to stitch_ndtiff.py (default: 1.0)")
    ap.add_argument("--channel", default="g", help="Channel passed to stitch_ndtiff.py (default: g)")
    return ap.parse_args()

def main():
    global BASE_DIR, SCALE, CHANNEL, MASTER_LOG_PATH

    args = parse_args()
    BASE_DIR = Path(args.base_dir)
    SCALE = args.scale
    CHANNEL = args.channel

    if not BASE_DIR.exists():
        print(f"Base folder does not exist: {BASE_DIR}")
        sys.exit(1)

    MASTER_LOG_PATH = BASE_DIR / "watcher.log"
    script_dir = Path(__file__).resolve().parent

    log_master(f"Watching (recursive) {BASE_DIR} (scripts in {script_dir}) | scale={SCALE} channel={CHANNEL}")

    while True:
        try:
            # Pick up datasets that (a) look like NDTiff roots, (b) have no dataset log yet
            for ds in list_candidate_datasets_recursive(BASE_DIR):
                # Only proceed when writer has finalized the dataset
                try:
                    d = Dataset(str(ds))
                    if not d.is_finished():
                        continue
                except Exception as e:
                    log_master(f"[{ds.name}] ndstorage open/read failed: {e}")
                    continue

                log_master(f"[{ds.name}] Finished detected; waiting {GRACE_AFTER_FINISH_SECONDS}s before processing.")
                time.sleep(GRACE_AFTER_FINISH_SECONDS)

                # Create the dataset log *now* (start of real processing). From here on,
                # presence of this file means "already handled" unless you delete it.
                log_ds(ds, "Processing started.")
                ok = run_processing(ds, script_dir)
                if ok:
                    log_ds(ds, "Processing complete.")
                else:
                    log_ds(ds, "Processing failed. Delete this log to retry.")

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Exiting on Ctrl+C")
            break
        except Exception as e:
            log_master(f"Top-level loop error: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
