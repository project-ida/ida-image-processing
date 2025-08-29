#!/usr/bin/env python3
"""
watch_and_process_ndtiff.py — event-driven (watchdog) or polling watcher for incoming NDTiff datasets.

Assumptions:
- ndstorage is installed; completeness check uses Dataset.is_finished() only.
- This script lives in the SAME folder as stitch_ndtiff.py and dzi_from_bigtiff.py.
- After a dataset is finished, we run:
    1) python stitch_ndtiff.py <dataset_folder> --scale 1.0 --channel g
    2) python dzi_from_bigtiff.py <dataset_folder>\\stitched --split-channel
- We avoid duplicate work via a sentinel file "auto_processed.ok" in the dataset folder.
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Set
from ndstorage import Dataset

# ------------------ User config ------------------
BASE_DIR       = r"C:\Users\MITLERN\leica-images"  # where new NDTiff datasets appear
POLL_SECONDS   = 30                                 # loop interval for is_finished() checks
SCALE          = "1.0"                              # passed to stitch_ndtiff.py
CHANNEL        = "g"                                # passed to stitch_ndtiff.py
INDEX_NAME     = "NDTiff.index"                     # identifies a dataset root
SENTINEL_DONE  = "auto_processed.ok"                # marks a dataset as processed
LOG_NAME       = "auto_process.log"                 # per-dataset log
# -------------------------------------------------

def log_line(ds_dir: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    try:
        (ds_dir / LOG_NAME).open("a", encoding="utf-8").write(line)
    except Exception:
        pass

def is_dataset_folder(p: Path) -> bool:
    return p.is_dir() and (p / INDEX_NAME).exists()

def find_unprocessed_datasets(base: Path):
    for child in base.iterdir():
        if is_dataset_folder(child) and not (child / SENTINEL_DONE).exists():
            yield child

def run_processing(ds_dir: Path, script_dir: Path) -> bool:
    stitch_script = script_dir / "stitch_ndtiff.py"
    dzi_script    = script_dir / "dzi_from_bigtiff.py"

    # 1) stitch_ndtiff.py <dataset> --scale 1.0 --channel g
    cmd1 = [sys.executable, str(stitch_script), str(ds_dir), "--scale", SCALE, "--channel", CHANNEL]
    log_line(ds_dir, f"Running: {' '.join(cmd1)}")
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError as e:
        log_line(ds_dir, f"ERROR: stitch_ndtiff failed with code {e.returncode}")
        return False

    # 2) dzi_from_bigtiff.py <dataset>\\stitched --split-channel
    stitched_dir = ds_dir / "stitched"
    cmd2 = [sys.executable, str(dzi_script), str(stitched_dir), "--split-channel"]
    log_line(ds_dir, f"Running: {' '.join(cmd2)}")
    try:
        subprocess.run(cmd2, check=True)
    except subprocess.CalledProcessError as e:
        log_line(ds_dir, f"ERROR: dzi_from_bigtiff failed with code {e.returncode}")
        return False

    return True

def main():
    base = Path(BASE_DIR)
    if not base.exists():
        print(f"Base folder does not exist: {base}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    print(f"Watching {base}... (scripts in {script_dir})")

    # Track candidate datasets (strings for set stability on Windows)
    pending: Set[str] = set()

    # Seed with any existing, unprocessed datasets
    for ds in find_unprocessed_datasets(base):
        pending.add(str(ds))

    # Try to use watchdog for immediate detection of new datasets
    use_watchdog = False
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        use_watchdog = True
    except Exception:
        print("watchdog not installed; will rely on periodic polling for new datasets.")

    observer = None
    if use_watchdog:
        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                p = Path(event.src_path)
                # If a folder was created, check if it's already a dataset
                if event.is_directory:
                    if is_dataset_folder(p) and not (p / SENTINEL_DONE).exists():
                        pending.add(str(p))
                else:
                    # If the index file appeared, add its parent folder
                    if p.name == INDEX_NAME:
                        ds = p.parent
                        if is_dataset_folder(ds) and not (ds / SENTINEL_DONE).exists():
                            pending.add(str(ds))

            def on_moved(self, event):
                # Handle renames that place INDEX_NAME or a dataset folder into BASE_DIR
                p = Path(event.dest_path)
                if p.is_dir():
                    if is_dataset_folder(p) and not (p / SENTINEL_DONE).exists():
                        pending.add(str(p))
                else:
                    if p.name == INDEX_NAME:
                        ds = p.parent
                        if is_dataset_folder(ds) and not (ds / SENTINEL_DONE).exists():
                            pending.add(str(ds))

        observer = Observer()
        observer.schedule(Handler(), str(base), recursive=False)
        observer.start()

    try:
        while True:
            # Pick up any datasets missed by events (or when watchdog is unavailable)
            for ds in find_unprocessed_datasets(base):
                pending.add(str(ds))

            # Check pending datasets for completion and process
            for ds_str in list(pending):
                ds = Path(ds_str)
                if not ds.exists():
                    pending.discard(ds_str)
                    continue
                if (ds / SENTINEL_DONE).exists():
                    pending.discard(ds_str)
                    continue

                log_line(ds, "Checking is_finished()...")
                try:
                    d = Dataset(str(ds))
                    if not d.is_finished():
                        log_line(ds, "Not finished yet; will check again.")
                        continue
                    log_line(ds, "Dataset is finished.")
                except Exception as e:
                    log_line(ds, f"ndstorage read failed: {e}")
                    continue

                ok = run_processing(ds, script_dir)
                if ok:
                    try:
                        (ds / SENTINEL_DONE).write_text("ok\n", encoding="utf-8")
                    except Exception:
                        pass
                    log_line(ds, "Processing complete. Sentinel written.")
                    pending.discard(ds_str)
                else:
                    log_line(ds, "Processing failed. Will retry later.")

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("Exiting on Ctrl+C")
    finally:
        if observer:
            observer.stop()
            observer.join()

if __name__ == "__main__":
    main()
