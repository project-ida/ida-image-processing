#!/usr/bin/env python3
"""
watch_and_process_ndtiff.py — simple polling watcher for incoming NDTiff datasets.

Debug-friendly behavior:
- It only processes datasets that DO NOT yet have an auto_process.log in them.
  → Delete auto_process.log to force a re-run for debugging.

Assumptions:
- ndstorage is installed; completeness check uses Dataset.is_finished() only.
- This script lives in the SAME folder as stitch_ndtiff.py and dzi_from_bigtiff.py.
- After a dataset is finished, we run:
    1) python stitch_ndtiff.py <dataset_folder> --scale 1.0 --channel g
    2) python dzi_from_bigtiff.py <dataset_folder>\\stitched --split-channel
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable
from ndstorage import Dataset

# ------------------ User config ------------------
BASE_DIR       = r"C:\Users\MITLERN\leica-images"  # where new NDTiff datasets appear
POLL_SECONDS   = 60                                 # check once per minute
GRACE_AFTER_FINISH_SECONDS = 60                     # wait 1 min after is_finished() before processing
SCALE          = "1.0"                              # passed to stitch_ndtiff.py
CHANNEL        = "g"                                # passed to stitch_ndtiff.py
INDEX_NAME     = "NDTiff.index"                     # identifies a dataset root
LOG_NAME       = "auto_process.log"                 # the per-dataset log (presence gates processing)

# Optional master log in BASE_DIR (set False to disable)
WRITE_MASTER_LOG  = True
MASTER_LOG_PATH   = Path(BASE_DIR) / "watcher.log"
# -------------------------------------------------

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

def is_dataset_folder(p: Path) -> bool:
    return p.is_dir() and (p / INDEX_NAME).exists()

def list_candidate_datasets(base: Path) -> Iterable[Path]:
    """Immediate subfolders with NDTiff.index and NO dataset log yet."""
    if not base.exists():
        return []
    for child in base.iterdir():
        if is_dataset_folder(child) and not (child / LOG_NAME).exists():
            yield child

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

def main():
    base = Path(BASE_DIR)
    if not base.exists():
        print(f"Base folder does not exist: {base}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    log_master(f"Watching {base} (scripts in {script_dir})")

    while True:
        try:
            # Pick up datasets that (a) look like NDTiff roots, (b) have no dataset log yet
            for ds in list_candidate_datasets(base):
                # Don't create the dataset log yet; we only create it when we're about to process
                # so that deleting it forces a re-run cleanly.
                try:
                    d = Dataset(str(ds))
                    if not d.is_finished():
                        # Keep quiet; we'll check again next cycle
                        continue
                except Exception as e:
                    # If we can't open it yet, just try again later
                    log_master(f"[{ds.name}] ndstorage open/read failed: {e}")
                    continue

                # Optional grace after finish(), to allow any last sync stragglers
                log_master(f"[{ds.name}] Finished detected; waiting {GRACE_AFTER_FINISH_SECONDS}s before processing.")
                time.sleep(GRACE_AFTER_FINISH_SECONDS)

                # Create the dataset log *now* (start of real processing). From here on,
                # presence of this file means "already handled" unless the user deletes it.
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
