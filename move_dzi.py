#!/usr/bin/env python3
"""
Move DZI pairs (file.dzi + file_files/) from an origin tree into a destination,
organized under a chosen subfolder name.

Subfolder naming priority:
1) --folder-name-override
2) --use-grandparent-folder-name  (uses grandparent directory of the .dzi file)
3) .dzi filename stem

Examples
--------
# Basic: move all DZI pairs found under ./in into ./out
python move_dzi.py --dzi-origin ./in --dzi-destination ./out

# Use grandparent folder as the subfolder name
python move_dzi.py --dzi-origin ./in --dzi-destination ./out --use-grandparent-folder-name

# Force a specific subfolder name
python move_dzi.py --dzi-origin ./in --dzi-destination ./out --folder-name-override SampleA

# Allow replacing existing destination files/dirs
python move_dzi.py --dzi-origin ./in --dzi-destination ./out --overwrite
"""

import argparse
import shutil
from pathlib import Path
import sys
import os

def find_files_folder(dzi_path: Path) -> Path | None:
    """
    Given /path/to/foo.dzi, look for a sibling directory named foo_files.
    If not found, try to find a matching *exact* directory anywhere under the same parent.
    """
    stem = dzi_path.stem
    candidate = dzi_path.parent / f"{stem}_files"
    if candidate.is_dir():
        return candidate

    # Fallback: search siblings under the same parent for exact match (handles odd cases)
    for p in dzi_path.parent.iterdir():
        if p.is_dir() and p.name == f"{stem}_files":
            return p

    return None

def compute_subfolder_name(dzi_path: Path, use_grandparent: bool, override: str | None) -> str:
    if override:
        return override
    if use_grandparent:
        gp = dzi_path.parent.parent
        return gp.name if gp and gp.name else dzi_path.stem
    return dzi_path.stem

def safe_move(src: Path, dst: Path, overwrite: bool) -> None:
    """
    Move src -> dst. If dst exists:
      - overwrite=False: raise FileExistsError
      - overwrite=True: remove dst then move
    """
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    # Ensure parent exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def move_one_pair(
    dzi_path: Path,
    dest_root: Path,
    use_grandparent: bool,
    override: str | None,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    """
    Move one DZI pair into a subfolder. Returns (target_dir, moved_dzi_path, moved_files_dir)
    """
    files_dir = find_files_folder(dzi_path)
    if files_dir is None:
        raise FileNotFoundError(
            f"Matching _files directory not found for {dzi_path} "
            f"(expected '{dzi_path.stem}_files' as a sibling)."
        )

    subfolder = compute_subfolder_name(dzi_path, use_grandparent, override)
    target_dir = dest_root / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    # Compute targets
    dzi_target = target_dir / dzi_path.name
    files_target = target_dir / files_dir.name

    # Move DZI first
    safe_move(dzi_path, dzi_target, overwrite=overwrite)
    # Then the folder
    safe_move(files_dir, files_target, overwrite=overwrite)

    return target_dir, dzi_target, files_target

def main():
    parser = argparse.ArgumentParser(description="Move DZI pairs into a destination folder.")
    parser.add_argument("--dzi-origin", required=True, type=Path,
                        help="Folder to search (recursively) for .dzi files.")
    parser.add_argument("--dzi-destination", required=True, type=Path,
                        help="Folder where DZI pairs will be moved.")
    parser.add_argument("--use-grandparent-folder-name", action="store_true",
                        help="Use the grandparent directory name of the .dzi as the subfolder name.")
    parser.add_argument("--folder-name-override", default=None,
                        help="Explicit subfolder name to use (overrides other naming).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing files/directories at the destination if they exist.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be moved without making changes.")
    args = parser.parse_args()

    origin: Path = args.dzi_origin
    destination: Path = args.dzi_destination

    if not origin.exists() or not origin.is_dir():
        print(f"ERROR: --dzi-origin does not exist or is not a directory: {origin}", file=sys.stderr)
        sys.exit(2)

    destination.mkdir(parents=True, exist_ok=True)

    dzi_list = sorted(origin.rglob("*.dzi"))
    if not dzi_list:
        print(f"No .dzi files found under: {origin}")
        sys.exit(0)

    print(f"Found {len(dzi_list)} .dzi file(s) under {origin}")
    moved_count = 0
    errors: list[str] = []

    for dzi in dzi_list:
        try:
            # Preview
            files_dir = find_files_folder(dzi)
            if files_dir is None:
                raise FileNotFoundError(
                    f"Missing _files folder for: {dzi} (expected '{dzi.stem}_files')"
                )
            subfolder = compute_subfolder_name(dzi, args.use_grandparent_folder_name, args.folder_name_override)
            target_dir = destination / subfolder
            dzi_target = target_dir / dzi.name
            files_target = target_dir / files_dir.name

            if args.dry_run:
                print(f"[DRY-RUN] Would move:")
                print(f"  DZI:    {dzi}  -> {dzi_target}")
                print(f"  FOLDER: {files_dir} -> {files_target}")
                continue

            target_dir, new_dzi, new_files = move_one_pair(
                dzi, destination, args.use_grandparent_folder_name, args.folder_name_override, args.overwrite
            )
            print(f"Moved: {dzi.name} and '{files_dir.name}' -> {target_dir}")
            moved_count += 1

        except Exception as e:
            msg = f"ERROR processing {dzi}: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if args.dry_run:
        print("Dry-run complete. No changes made.")
    else:
        print(f"Done. Successfully moved {moved_count} DZI pair(s).")
        if errors:
            print(f"{len(errors)} error(s) occurred:", file=sys.stderr)
            for m in errors:
                print(" - " + m, file=sys.stderr)

if __name__ == "__main__":
    # On Linux/macOS make executable: chmod +x move_dzi.py
    # Then run as shown in the docstring examples.
    main()
