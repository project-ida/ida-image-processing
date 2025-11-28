#!/usr/bin/env python3

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional
from ndstorage import Dataset

# ---------- fixed config ----------
POLL_SECONDS   = 60                      # check once per minute
GRACE_AFTER_FINISH_SECONDS = 5           # wait after is_finished() before processing
INDEX_NAME     = "NDTiff.index"          # identifies a dataset root
LOG_NAME       = "auto_process.log"      # per-dataset log (presence gates processing)
WRITE_MASTER_LOG = True                  # also keep one global log in BASE_DIR
# ----------------------------------

# filled from CLI:
BASE_DIR: Path
SCALE: str
CHANNEL: str
MASTER_LOG_PATH: Path
DZI_DEST: Optional[Path]

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
    Recursively yield dataset roots: folders with NDTiff.index and without auto_process.log.
    """
    if not base.exists():
        return []
    for root, dirs, files in os.walk(base):
        if INDEX_NAME in files and LOG_NAME not in files:
            yield Path(root)

def read_rotation_arg(ds_dir: Path) -> list[str]:
    """
    If <ds_dir>/rotation.txt exists, read the first line as a float (degrees).
    Return ["--rotate", "<value>"] on success, else [].
    """
    rot_path = ds_dir / "rotation.txt"
    if not rot_path.exists():
        return []
    try:
        first_line = rot_path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        val = -float(first_line)
        return ["--rotate", str(val)]
    except Exception as e:
        log_ds(ds_dir, f"rotation.txt present but not usable ({e}); continuing without rotation.")
        return []

def run_processing(ds_dir: Path, script_dir: Path, dzi_destination_dir: Optional[Path] = None) -> bool:
    """Run the steps; return True on success."""
    stitch_script = script_dir / "stitch_ndtiff.py"
    dzi_script    = script_dir / "dzi_from_bigtiff.py"
    moving_script = script_dir / "move_dzi.py"

    # Optional rotation argument from rotation.txt
    rotate_args = read_rotation_arg(ds_dir)
    if rotate_args:
        log_ds(ds_dir, f"Using rotation from rotation.txt: {rotate_args[1]} degrees")

    # Step 1: stitch
    cmd1 = [sys.executable, str(stitch_script), str(ds_dir),
            "--scale", SCALE, "--channel", CHANNEL, *rotate_args, "--overwrite"]
    log_ds(ds_dir, f"Running: {' '.join(cmd1)}")
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError as e:
        log_ds(ds_dir, f"ERROR: stitch_ndtiff failed with code {e.returncode}")
        return False

    # Step 2: DZI from stitched
    stitched_dir = ds_dir / "stitched"
    cmd2 = [sys.executable, str(dzi_script), str(stitched_dir), "--split-channel", "--no-skip-if-exists"]
    log_ds(ds_dir, f"Running: {' '.join(cmd2)}")
    try:
        subprocess.run(cmd2, check=True)
    except subprocess.CalledProcessError as e:
        log_ds(ds_dir, f"ERROR: dzi_from_bigtiff failed with code {e.returncode}")
        return False

    # Step 3: move DZIs (optional)
    if dzi_destination_dir is not None:
        dzi_origin_dir = stitched_dir
        cmd3 = [
            sys.executable, str(moving_script),
            "--dzi-origin", str(dzi_origin_dir),
            "--dzi-destination", str(dzi_destination_dir),
            "--use-greatgrandparent-folder-name",
            "--overwrite",
        ]
        log_ds(ds_dir, f"Running: {' '.join(cmd3)}")
        try:
            subprocess.run(cmd3, check=True)
        except subprocess.CalledProcessError as e:
            log_ds(ds_dir, f"ERROR: move_dzi failed with code {e.returncode}")
            return False

    return True

def parse_args():
    ap = argparse.ArgumentParser(description="Watch (recursively) for finished NDTiff datasets and process them.")
    ap.add_argument("--base-dir", required=True, help="Root folder to scan recursively for NDTiff datasets")
    ap.add_argument("--scale", default="1.0", help="Scale passed to stitch_ndtiff.py (default: 1.0)")
    ap.add_argument("--channel", default="g", help="Channel passed to stitch_ndtiff.py (default: g)")
    ap.add_argument("--dzi-destination", required=False, help="Optional destination folder for DZIs")
    return ap.parse_args()

def main():
    global BASE_DIR, SCALE, CHANNEL, DZI_DEST, MASTER_LOG_PATH

    args = parse_args()
    BASE_DIR = Path(args.base_dir)
    SCALE = args.scale
    CHANNEL = args.channel
    DZI_DEST = Path(args.dzi_destination) if args.dzi_destination else None

    if not BASE_DIR.exists():
        print(f"Base folder does not exist: {BASE_DIR}")
        sys.exit(1)

    MASTER_LOG_PATH = BASE_DIR / "watcher.log"
    script_dir = Path(__file__).resolve().parent

    log_master(
        f"Watching (recursive) {BASE_DIR} (scripts in {script_dir}) | "
        f"scale={SCALE} channel={CHANNEL} dzi-destination={DZI_DEST}"
    )

    while True:
        try:
            for ds in list_candidate_datasets_recursive(BASE_DIR):
                # Try to open and verify finished; any failure → skip, we'll retry next minute
                try:
                    d = Dataset(str(ds))
                except Exception as e:
                    log_master(f"[{ds.name}] Dataset open failed ({e.__class__.__name__}: {e}); retry later.")
                    continue

                log_master(f"[{ds.name}] Finished detected; waiting {GRACE_AFTER_FINISH_SECONDS}s before processing.")
                time.sleep(GRACE_AFTER_FINISH_SECONDS)

                # Start processing (creating the dataset log now enables the 'delete to retry' behavior)
                log_ds(ds, "Processing started.")
                ok = run_processing(ds, script_dir, DZI_DEST)
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
