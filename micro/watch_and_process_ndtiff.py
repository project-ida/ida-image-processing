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
POLL_SECONDS   = 60
GRACE_AFTER_FINISH_SECONDS = 5
INDEX_NAME     = "NDTiff.index"
LOG_NAME       = "auto_process.log"           # Real run
LOG_NAME_DRY   = "auto_process.dryrun.log"    # Dry-run only
WRITE_MASTER_LOG = True
# ----------------------------------

# filled from CLI:
BASE_DIR: Path
SCALE: str
CHANNEL: str
MASTER_LOG_PATH: Path
DZI_DEST: Optional[Path]
DRY_RUN: bool = False

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

def log_ds(ds_dir: Path, msg: str, dry_run: bool = False) -> None:
    log_filename = LOG_NAME_DRY if dry_run else LOG_NAME
    line = f"[{ts()}][{ds_dir.name}] {msg}\n"
    print(line, end="")
    try:
        (ds_dir / log_filename).open("a", encoding="utf-8").write(line)
    except Exception:
        pass
    if WRITE_MASTER_LOG:
        try:
            MASTER_LOG_PATH.open("a", encoding="utf-8").write(line)
        except Exception:
            pass

def list_candidate_datasets_recursive(base: Path, dry_run: bool = False) -> Iterable[Path]:
    """
    Yield dataset roots with NDTiff.index.

    - Normal mode: skip if auto_process.log exists
    - Dry-run mode: skip if EITHER auto_process.log OR auto_process.dryrun.log exists
    """
    if not base.exists():
        return

    for root, dirs, files in os.walk(base):
        if INDEX_NAME not in files:
            continue

        # Real run: only check real log
        if not dry_run:
            if LOG_NAME not in files:
                yield Path(root)
        # Dry-run: skip if ANY log exists
        else:
            if LOG_NAME not in files and LOG_NAME_DRY not in files:
                yield Path(root)

def run_processing(ds_dir: Path, script_dir: Path, dzi_destination_dir: Optional[Path] = None, dry_run: bool = False) -> bool:
    stitch_script = script_dir / "stitch_ndtiff.py"
    dzi_script    = script_dir.parent / "dzi_from_bigtiff.py"
    moving_script = script_dir.parent / "move_dzi.py"

    # Step 1: stitch
    cmd1 = [sys.executable, str(stitch_script), str(ds_dir), "--scale", SCALE, "--channel", CHANNEL]
    log_msg = f"Would run: {' '.join(cmd1)}" if dry_run else f"Running: {' '.join(cmd1)}"
    log_ds(ds_dir, log_msg, dry_run=dry_run)
    if not dry_run:
        try:
            subprocess.run(cmd1, check=True)
        except subprocess.CalledProcessError as e:
            log_ds(ds_dir, f"ERROR: stitch_ndtiff failed with code {e.returncode}", dry_run=dry_run)
            return False
    else:
        log_ds(ds_dir, "DRY-RUN: Skipping execution of stitch_ndtiff.py", dry_run=dry_run)

    # Step 2: DZI from stitched
    stitched_dir = ds_dir / "stitched"
    cmd2 = [sys.executable, str(dzi_script), str(stitched_dir), "--split-channel"]
    log_msg = f"Would run: {' '.join(cmd2)}" if dry_run else f"Running: {' '.join(cmd2)}"
    log_ds(ds_dir, log_msg, dry_run=dry_run)
    if not dry_run:
        try:
            subprocess.run(cmd2, check=True)
        except subprocess.CalledProcessError as e:
            log_ds(ds_dir, f"ERROR: dzi_from_bigtiff failed with code {e.returncode}", dry_run=dry_run)
            return False
    else:
        log_ds(ds_dir, "DRY-RUN: Skipping execution of dzi_from_bigtiff.py", dry_run=dry_run)

    # Step 3: move DZIs
    if dzi_destination_dir is not None:
        cmd3 = [
            sys.executable, str(moving_script),
            "--dzi-origin", str(stitched_dir),
            "--dzi-destination", str(dzi_destination_dir),
            "--use-greatgrandparent-folder-name"
        ]
        log_msg = f"Would run: {' '.join(cmd3)}" if dry_run else f"Running: {' '.join(cmd3)}"
        log_ds(ds_dir, log_msg, dry_run=dry_run)
        if not dry_run:
            try:
                subprocess.run(cmd3, check=True)
            except subprocess.CalledProcessError as e:
                log_ds(ds_dir, f"ERROR: move_dzi failed with code {e.returncode}", dry_run=dry_run)
                return False
        else:
            log_ds(ds_dir, "DRY-RUN: Skipping execution of move_dzi.py", dry_run=dry_run)

    return True

def parse_args():
    ap = argparse.ArgumentParser(description="Watch (recursively) for finished NDTiff datasets and process them.")
    ap.add_argument("--base-dir", required=True, help="Root folder to scan recursively for NDTiff datasets")
    ap.add_argument("--scale", default="1.0", help="Scale passed to stitch_ndtiff.py (default: 1.0)")
    ap.add_argument("--channel", default="g", help="Channel passed to stitch_ndtiff.py (default: g)")
    ap.add_argument("--dzi-destination", required=False, help="Optional destination folder for DZIs")
    ap.add_argument("--dry-run", action="store_true", help="Preview only. Skips if any log (real or dry) exists.")
    return ap.parse_args()

def main():
    global BASE_DIR, SCALE, CHANNEL, DZI_DEST, MASTER_LOG_PATH, DRY_RUN

    args = parse_args()
    BASE_DIR = Path(args.base_dir)
    SCALE = args.scale
    CHANNEL = args.channel
    DZI_DEST = Path(args.dzi_destination) if args.dzi_destination else None
    DRY_RUN = args.dry_run

    if not BASE_DIR.exists():
        print(f"Base folder does not exist: {BASE_DIR}")
        sys.exit(1)

    MASTER_LOG_PATH = BASE_DIR / "watcher.log"
    script_dir = Path(__file__).resolve().parent

    mode = " (DRY-RUN)" if DRY_RUN else ""
    log_master(f"Watching (recursive) {BASE_DIR} (scripts in {script_dir}) | scale={SCALE} channel={CHANNEL} dzi-destination={DZI_DEST}{mode}")

    while True:
        try:
            for ds in list_candidate_datasets_recursive(BASE_DIR, dry_run=DRY_RUN):
                try:
                    d = Dataset(str(ds))
                except Exception as e:
                    log_master(f"[{ds.name}] Dataset open failed ({e.__class__.__name__}: {e}); retry later.")
                    continue

                log_master(f"[{ds.name}] Finished detected; waiting {GRACE_AFTER_FINISH_SECONDS}s before processing.")
                time.sleep(GRACE_AFTER_FINISH_SECONDS)

                log_ds(ds, "Processing started.", dry_run=DRY_RUN)
                ok = run_processing(ds, script_dir, DZI_DEST, dry_run=DRY_RUN)
                if ok:
                    log_ds(ds, "Processing complete." + (" (DRY-RUN)" if DRY_RUN else ""), dry_run=DRY_RUN)
                else:
                    log_ds(ds, "Processing failed. Delete this log to retry.", dry_run=DRY_RUN)

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Exiting on Ctrl+C")
            break
        except Exception as e:
            log_master(f"Top-level loop error: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
