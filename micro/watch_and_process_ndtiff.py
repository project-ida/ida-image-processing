#!/usr/bin/env python3

import os
import sys
import time
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional
from ndstorage import Dataset
import traceback

# ---------- fixed config ----------
POLL_SECONDS   = 60                      # check once per minute
GRACE_AFTER_FINISH_SECONDS = 5           # wait after is_finished() before processing
INDEX_NAME     = "NDTiff.index"          # identifies a dataset root
TRIGGER_NAME   = "ready_to_process.txt"   # when this file is present, the processing starts

# Logs:
LOG_NAME       = "auto_process.log"           # Real run
LOG_NAME_DRY   = "auto_process.dryrun.log"    # Dry-run only
WRITE_MASTER_LOG = True                       # also keep one global log in BASE_DIR
# ----------------------------------

# filled from CLI:
BASE_DIR: Path
SCALE: str
CHANNEL: str
MASTER_LOG_PATH: Path
DZI_DEST: Optional[Path]
DRY_RUN: bool = False   # global dry-run flag


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
    """
    Per-dataset log.
    - Real run:  auto_process.log
    - Dry-run:   auto_process.dryrun.log
    """
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
    Yield dataset roots that contain TRIGGER_NAME (ready_to_process.txt).

    - Normal mode: skip if auto_process.log exists
    - Dry-run mode: skip if EITHER auto_process.log OR auto_process.dryrun.log exists
    - If TRIGGER_NAME contains a line like "number_of_files=<N>", only yield once at least
      N .tif files exist in that same directory.
    """
    if not base.exists():
        return

    for root, dirs, files in os.walk(base):
        if TRIGGER_NAME not in files:
            continue

        # Log rules (skip already-processed datasets)
        # Real run: only check real log
        if not dry_run:
            if LOG_NAME in files:
                continue
        # Dry-run: skip if ANY log exists
        else:
            if LOG_NAME in files or LOG_NAME_DRY in files:
                continue

        ds_dir = Path(root)
        trigger_path = ds_dir / TRIGGER_NAME

        # If trigger file is empty: existence triggers immediately.
        # If it contains "number_of_files=<N>": wait until at least N files exist.
        try:
            trigger_text = trigger_path.read_text(encoding="utf-8")
        except Exception:
            # If we can't read it (e.g., syncing/incomplete), retry next poll.
            continue

        if trigger_text.strip() != "":
            expected: Optional[int] = None
            for line in trigger_text.splitlines():
                if not re.match(r"^\s*number_of_files\s*=", line, re.IGNORECASE):
                    continue
                value = line.split("=", 1)[1].split("#", 1)[0].strip()
                if value == "":
                    # Key present but value missing; treat as "not ready yet".
                    expected = -1
                    break
                try:
                    expected = int(value)
                except ValueError:
                    expected = -1
                break

            if expected is not None:
                if expected < 0:
                    continue
                if expected > 0:
                    present_tifs = sum(1 for f in files if f.lower().endswith((".tif", ".tiff")))
                    if present_tifs < expected:
                        continue

        yield ds_dir


def _parse_numeric_expr(expr: str) -> float:
    """
    Parse a numeric value or simple arithmetic expression safely.

    Supports examples like:
        1.0
        -3.5
        -90+1.45
        0.5-0.1
        1*1.007
        (1-0.002)*1.01

    Only digits, +, -, *, /, decimal points, e/E, parentheses, and spaces
    are allowed. Uses eval() with builtins disabled.
    """
    expr = expr.strip()
    # first try direct float
    try:
        return float(expr)
    except ValueError:
        allowed = set("0123456789.+-eE*/() ")
        if not expr or any(ch not in allowed for ch in expr):
            raise ValueError(f"Unsupported numeric expression: {expr!r}")
        val = eval(expr, {"__builtins__": None}, {})
        return float(val)


def read_transform_args(ds_dir: Path, dry_run: bool = False) -> list[str]:
    """
    Read rotation and scale from <ds_dir>/config.txt if present.

      rotation=<float or simple expression>   (degrees, CCW)
      scale=<float or simple expression>      (dimensionless)

    Returns a list of CLI args like:
      ["--scale", "<scale>", "--rotate", "<rotation>"]

    - If scale is missing in config.txt, we fall back to the watcher
      CLI --scale value.
    - If rotation is missing, we simply omit --rotate.
    """
    cfg_path = ds_dir / "config.txt"
    rotation_val = None
    scale_val = None

    if cfg_path.exists():
        try:
            text = cfg_path.read_text(encoding="utf-8")

            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "rotation":
                    rotation_val = _parse_numeric_expr(value)
                elif key == "scale":
                    scale_val = _parse_numeric_expr(value)

            if rotation_val is not None:
                log_ds(
                    ds_dir,
                    f"Using rotation from config.txt: {rotation_val} degrees",
                    dry_run=dry_run,
                )

            if scale_val is not None:
                log_ds(
                    ds_dir,
                    f"Using scale from config.txt: {scale_val}",
                    dry_run=dry_run,
                )

        except Exception as e:
            log_ds(
                ds_dir,
                f"config.txt present but transform not fully usable ({e}); "
                f"falling back where possible.",
                dry_run=dry_run,
            )

    # Fallback for scale: watcher CLI --scale
    if scale_val is None:
        try:
            scale_val = float(SCALE)
            log_ds(
                ds_dir,
                f"No scale in config.txt; using watcher CLI scale={scale_val}",
                dry_run=dry_run,
            )
        except Exception:
            # Last resort, default 1.0
            scale_val = 1.0
            log_ds(
                ds_dir,
                "No usable scale from config.txt or CLI; defaulting to 1.0",
                dry_run=dry_run,
            )

    args: list[str] = []

    # Always pass scale (so downstream scripts can rely on it)
    args.extend(["--scale", str(scale_val)])

    # Only pass rotation if we actually have one
    if rotation_val is not None:
        args.extend(["--rotate", str(rotation_val)])

    return args


def run_processing(
    ds_dir: Path,
    script_dir: Path,
    dzi_destination_dir: Optional[Path] = None,
    dry_run: bool = False
) -> bool:
    """
    Run the steps; return True on success.

    - Respects dry_run: logs "Would run:" and does not execute subprocesses.
    - Includes:
        * rotation/scale via config.txt (falls back to watcher --scale)
        * --overwrite to stitch_ndtiff.py
        * --split-channel and --no-skip-if-exists to dzi_from_bigtiff.py
        * --use-greatgrandparent-folder-name and --overwrite to move_dzi.py
    """
    # Layout:
    #   watcher + stitch_ndtiff.py live in the same folder (script_dir)
    #   dzi_from_bigtiff.py and move_dzi.py live in the parent folder
    stitch_script = script_dir / "stitch_ndtiff.py"
    dzi_script    = script_dir.parent / "dzi_from_bigtiff.py"
    metric_grid_script    = script_dir.parent / "create_metric_grid.py"
    stitching_grid_script = script_dir.parent / "create_stitching_grid.py"
    moving_script = script_dir.parent / "move_dzi.py"

    # Rotation + scale arguments from config.txt / watcher CLI
    transform_args = read_transform_args(ds_dir, dry_run=dry_run)

    # Step 1: stitch
    cmd1 = [
        sys.executable, str(stitch_script), str(ds_dir),
        *transform_args,
        "--channel", CHANNEL,
        "--overwrite",  # allow stitched BigTIFF overwrite
    ]
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
    cmd2 = [
        sys.executable, str(dzi_script),
        str(stitched_dir),
        "--split-channel",
        "--no-skip-if-exists",   # force regeneration of DZIs
    ]
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

    # Step 2.5: create overlay grids
    # metric_grid
    cmd_metric = [
        sys.executable, str(metric_grid_script),
        str(ds_dir)
    ]
    log_msg = f"Would run: {' '.join(cmd_metric)}" if dry_run else f"Running: {' '.join(cmd_metric)}"
    log_ds(ds_dir, log_msg, dry_run=dry_run)
    if not dry_run:
        try:
            subprocess.run(cmd_metric, check=True)
        except subprocess.CalledProcessError as e:
            log_ds(ds_dir, f"ERROR: create_metric_grid failed with code {e.returncode}", dry_run=dry_run)
    else:
        log_ds(ds_dir, "DRY-RUN: Skipping execution of create_metric_grid.py", dry_run=dry_run)

    # stitching_grid
    cmd_stitching = [
        sys.executable, str(stitching_grid_script),
        str(ds_dir),
        *transform_args,
    ]
    log_msg = f"Would run: {' '.join(cmd_stitching)}" if dry_run else f"Running: {' '.join(cmd_stitching)}"
    log_ds(ds_dir, log_msg, dry_run=dry_run)
    if not dry_run:
        try:
            subprocess.run(cmd_stitching, check=True)
        except subprocess.CalledProcessError as e:
            log_ds(ds_dir, f"ERROR: create_stitching_grid failed with code {e.returncode}", dry_run=dry_run)
    else:
        log_ds(ds_dir, "DRY-RUN: Skipping execution of create_stitching_grid.py", dry_run=dry_run)

    # Step 3: move DZIs
    if dzi_destination_dir is not None:
        cmd3 = [
            sys.executable, str(moving_script),
            "--dzi-origin", str(stitched_dir),
            "--dzi-destination", str(dzi_destination_dir),
            "--use-greatgrandparent-folder-name",
            "--overwrite",  # allow replacing existing DZI targets
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
    ap = argparse.ArgumentParser(
        description="Watch (recursively) for finished NDTiff datasets and process them."
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Root folder to scan recursively for NDTiff datasets"
    )
    ap.add_argument(
        "--scale", default="1.0",
        help="Default scale passed to stitch_ndtiff.py if config.txt has no 'scale=' (default: 1.0)"
    )
    ap.add_argument(
        "--channel", default="g",
        help="Channel passed to stitch_ndtiff.py (default: g)"
    )
    ap.add_argument(
        "--dzi-destination", required=False,
        help="Optional destination folder for DZIs"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Preview only. Skips if any log (real or dry) exists and does not run subprocesses."
    )
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
    log_master(
        f"Watching (recursive) {BASE_DIR} (scripts in {script_dir}) | "
        f"default_scale={SCALE} channel={CHANNEL} dzi-destination={DZI_DEST}{mode}"
    )

    while True:
        try:
            for ds in list_candidate_datasets_recursive(BASE_DIR, dry_run=DRY_RUN):
                # Try to open and verify finished; any failure → skip, we'll retry next minute
                try:
                    _ = Dataset(str(ds))
                except Exception as e:
                    log_master(f"[{ds.name}] Dataset open failed ({e.__class__.__name__}: {e}); retry later.")
                    continue

                log_master(f"[{ds.name}] Finished detected; waiting {GRACE_AFTER_FINISH_SECONDS}s before processing.")
                time.sleep(GRACE_AFTER_FINISH_SECONDS)

                # Start processing (creating the dataset log now enables the 'delete to retry' behavior)
                log_ds(ds, "Processing started.", dry_run=DRY_RUN)
                ok = run_processing(ds, script_dir, DZI_DEST, dry_run=DRY_RUN)
                if ok:
                    suffix = " (DRY-RUN)" if DRY_RUN else ""
                    log_ds(ds, f"Processing complete.{suffix}", dry_run=DRY_RUN)
                else:
                    log_ds(ds, "Processing failed. Delete this log to retry.", dry_run=DRY_RUN)

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Exiting on Ctrl+C")
            break
        except Exception as e:
            #log_master(f"Top-level loop error: {e}")
            log_master("Top-level loop error:\n" + traceback.format_exc())
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
