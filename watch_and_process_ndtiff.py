# Let's create a Python watcher script that polls a base folder for new NDTiff datasets,
# determines when they're complete (via either ndstorage.is_finished() or a quiescence window),
# then runs two command-line scripts in sequence. It logs actions and avoids duplicate work.

from pathlib import Path
import time
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

SCRIPT_TEXT = r'''#!/usr/bin/env python3
"""
watch_and_process_ndtiff.py

Watches a base folder for new NDTiff datasets. When a dataset is complete,
runs the two processing steps shown in the screenshot:
  1) stitch_ndtiff.py <dataset_folder> --scale=1.0 --channel g
  2) dzi_from_bigtiff.py <dataset_folder>\stitched --split-channel

How "complete" is detected (in this order):
  A) If 'ndstorage' is installed, try to open the dataset and check is_finished().
  B) Otherwise, require a quiescence window: no file size changes for QUIET_SECONDS.

Also supports a "grace" time after the folder first appears, to give the writer
time to initialize.

Creates a sentinel file 'auto_processed.ok' inside each dataset folder after success
to avoid duplicate work. Logs to 'auto_process.log' in the dataset folder.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

# =====================
# USER CONFIG (EDIT ME)
# =====================
# Base folder to watch for incoming datasets
BASE_DIR = r"C:\Users\MITLERN\leica-images"

# Paths to your processing scripts
STITCH_SCRIPT = r"C:\Users\MITLERN\ida-image-processing\stitch_ndtiff.py"
DZI_SCRIPT    = r"C:\Users\MITLERN\ida-image-processing\dzi_from_bigtiff.py"

# Parameters for stitch_ndtiff.py
SCALE = "1.0"
CHANNEL = "g"   # e.g. "g", or "r", or whatever your script expects

# Quiescence and polling settings
POLL_SECONDS   = 30           # how often to rescan BASE_DIR
QUIET_SECONDS  = 180          # how long files must remain unchanged to consider dataset complete
MIN_AGE_SECONDS = 60          # ignore folders younger than this (give the writer a moment to start)
MAX_PARALLEL = 1              # process at most N datasets at the same time

# File/Folder names that indicate an NDTiff dataset
INDEX_NAME = "NDTiff.index"   # presence of this file inside a folder indicates a dataset root

# =====================
# END USER CONFIG
# =====================

SENTINEL_DONE = "auto_processed.ok"
LOG_NAME = "auto_process.log"

def log_line(dataset_dir: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    try:
        with open(dataset_dir / LOG_NAME, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

def find_candidate_datasets(base: Path) -> List[Path]:
    """Return folders containing INDEX_NAME and not already processed."""
    candidates = []
    if not base.exists():
        return candidates
    for p in base.iterdir():
        if p.is_dir():
            index_path = p / INDEX_NAME
            if index_path.exists():
                # skip if sentinel exists
                if (p / SENTINEL_DONE).exists():
                    continue
                # skip if folder is too young
                age_seconds = (time.time() - p.stat().st_mtime)
                if age_seconds < MIN_AGE_SECONDS:
                    continue
                candidates.append(p)
    return candidates

def folder_snapshot(folder: Path) -> Dict[str, Tuple[int, float]]:
    """
    Create a snapshot: {relative_path: (size, mtime)}
    """
    snap = {}
    for root, dirs, files in os.walk(folder):
        for name in files:
            fp = Path(root) / name
            try:
                st = fp.stat()
            except FileNotFoundError:
                continue
            rel = str(fp.relative_to(folder))
            snap[rel] = (st.st_size, st.st_mtime)
    return snap

def is_quiet(folder: Path, quiet_seconds: int) -> bool:
    """Return True if folder contents (file count/sizes/mtimes) are unchanged over quiet_seconds."""
    snap1 = folder_snapshot(folder)
    time.sleep(quiet_seconds)
    snap2 = folder_snapshot(folder)
    return snap1 == snap2

def is_finished_via_ndstorage(folder: Path) -> bool:
    """Try to use ndstorage.Dataset.is_finished() if available. Fallback returns False."""
    try:
        from ndstorage import Dataset
    except Exception:
        return False
    try:
        d = Dataset(str(folder))
        return bool(d.is_finished())
    except Exception:
        return False

def run_processing(dataset_dir: Path) -> bool:
    """Run the two processing steps. Returns True on success."""
    # 1) stitch_ndtiff.py <dataset> --scale=1.0 --channel g
    cmd1 = [sys.executable, STITCH_SCRIPT, str(dataset_dir), "--scale", SCALE, "--channel", CHANNEL]
    log_line(dataset_dir, f"Running: {' '.join(cmd1)}")
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError as e:
        log_line(dataset_dir, f"ERROR: stitch_ndtiff failed with code {e.returncode}")
        return False

    # 2) dzi_from_bigtiff.py <dataset>\stitched --split-channel
    stitched_dir = dataset_dir / "stitched"
    cmd2 = [sys.executable, DZI_SCRIPT, str(stitched_dir), "--split-channel"]
    log_line(dataset_dir, f"Running: {' '.join(cmd2)}")
    try:
        subprocess.run(cmd2, check=True)
    except subprocess.CalledProcessError as e:
        log_line(dataset_dir, f"ERROR: dzi_from_bigtiff failed with code {e.returncode}")
        return False

    return True

def main():
    base = Path(BASE_DIR)
    if not base.exists():
        print(f"Base folder does not exist: {base}")
        sys.exit(1)

    processing_now: List[Path] = []

    print(f"Watching {base} for new NDTiff datasets...")
    while True:
        try:
            # Clean up finished jobs from the in-progress list
            processing_now = [p for p in processing_now if not (p / SENTINEL_DONE).exists()]

            candidates = find_candidate_datasets(base)
            # Prioritize oldest first (by mtime) to be fair
            candidates.sort(key=lambda p: p.stat().st_mtime)

            for ds in candidates:
                if len(processing_now) >= MAX_PARALLEL:
                    break
                # Skip if we already started processing but sentinel not yet created
                if ds in processing_now:
                    continue

                log_line(ds, f"Candidate dataset detected")

                # Check completeness
                finished = is_finished_via_ndstorage(ds)
                if finished:
                    log_line(ds, "Dataset reports is_finished() via ndstorage.")
                else:
                    log_line(ds, f"ndstorage check unavailable or incomplete; verifying quiescence for {QUIET_SECONDS}s...")
                    if not is_quiet(ds, QUIET_SECONDS):
                        log_line(ds, "Still changing; will check again later.")
                        continue
                    log_line(ds, "Quiescence window passed. Treating dataset as complete.")

                # Run processing
                ok = run_processing(ds)
                if ok:
                    try:
                        (ds / SENTINEL_DONE).write_text("ok\n", encoding="utf-8")
                    except Exception:
                        pass
                    log_line(ds, "Processing complete. Sentinel written.")
                else:
                    log_line(ds, "Processing failed. Will retry later after next quiescence.")

                processing_now.append(ds)

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Exiting on Ctrl+C")
            break
        except Exception as e:
            # Avoid dying on transient errors
            print(f"[{datetime.now().isoformat()}] Top-level loop error: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
'''

# Write the script to a file for the user
output_path = Path("/mnt/data/watch_and_process_ndtiff.py")
output_path.write_text(SCRIPT_TEXT, encoding="utf-8")
output_path
