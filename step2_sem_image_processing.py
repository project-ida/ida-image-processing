#!/usr/bin/env python3
"""
Step 2 — SEM image → PNG tiles + Fiji TileConfiguration.txt

Inputs
------
- --sem-folder: directory containing files like "<base>_sem.npz"
  Each NPZ must contain a 2D array under key "sem_data". If that key
  is missing, the first array in the NPZ is used as a fallback.
- --metadata-folder: directory with "<base>_metadata.txt" (simple key:value or key=value lines)
  Must include stage positions, ideally:
      "/1/EDS/Header/Stage Position/X"
      "/1/EDS/Header/Stage Position/Y"
  Units are typically mm — this script treats them as mm and converts to µm.

What it does
------------
1) Converts each SEM NPZ to an 8-bit PNG (grayscale).
   Normalization options:
     - auto        : per-image min..max
     - absolute16  : 0..65535 → 0..255
     - absolute    : 0..<dtype max> → 0..255
     - fixed       : use --vmin/--vmax
2) Builds a Fiji "TileConfiguration.txt" using stage positions:
   - Read stage X/Y (assumed mm → µm)
   - Optional flips with --invert-x / --invert-y (applied *before* normalization)
   - Subtract global mins so the top-left tile starts at (0,0)
   - Convert µm → px with --um-per-px (default 0.5490099 µm/px)
   - Writes pixel coordinates as floats

Outputs
-------
- <outdir>/sem-images-png/<base>_sem.png
- <outdir>/TileConfiguration.txt   (uses basenames of PNGs)

Notes
-----
- If metadata is missing, the image is still written; it is excluded
  from TileConfiguration (a warning is printed).
- Keep TileConfiguration.txt and the PNG tiles in the same folder
  (or adjust Fiji's working directory) so Fiji can resolve the filenames.
"""

from __future__ import annotations
import argparse
import sys
import re
from pathlib import Path

import numpy as np
from PIL import Image

# -------------------------- Metadata helpers --------------------------

def parse_metadata_txt(path: Path) -> dict[str, str]:
    """
    Accepts both 'key = value' and 'key: value' lines.
    """
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if "=" in line or ":" in line:
            try:
                k, v = re.split(r"\s*[=:]\s*", line, maxsplit=1)
                meta[k.strip()] = v.strip()
            except Exception:
                continue
    return meta

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def meta_num(meta: dict[str, str], key_suffix: str, default: float | None = None) -> float | None:
    """
    Find first key whose name ends with key_suffix (case-insensitive) and
    return the first numeric token from its value as float.
    """
    ksuf = key_suffix.lower()
    for k, v in meta.items():
        if k.lower().endswith(ksuf):
            m = _num_re.search(v)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    pass
    return default

# ---------------------------- Image I/O --------------------------------

def load_sem_npz(npz_path: Path) -> np.ndarray:
    """
    Load 2D SEM image array from NPZ.
    Prefer 'sem_data' but fall back to first array key.
    """
    with np.load(npz_path) as z:
        if "sem_data" in z:
            arr = z["sem_data"]
        else:
            arr = z[z.files[0]]
    if arr.ndim != 2:
        raise ValueError(f"{npz_path.name}: expected 2D array, got shape {arr.shape}")
    return arr

def to_uint8(arr: np.ndarray,
             method: str = "auto",
             vmin: float | None = None,
             vmax: float | None = None) -> np.ndarray:
    """
    Normalize a numeric 2D array to uint8 [0,255].
    """
    a = arr.astype(np.float64, copy=False)

    if method == "auto":
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    elif method == "fixed":
        if vmin is None or vmax is None or vmax <= vmin:
            raise ValueError("--norm fixed requires valid --vmin < --vmax")
        lo, hi = float(vmin), float(vmax)
    elif method == "absolute16":
        lo, hi = 0.0, 65535.0
    elif method == "absolute":
        # use dtype range if integer, else fall back to observed range
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            lo, hi = float(info.min), float(info.max)
        else:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    else:
        raise ValueError(f"Unknown --norm '{method}'")

    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)

    a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)

# ----------------------- TileConfiguration.txt -------------------------

def write_tileconfig(out_path: Path, entries: list[tuple[str, float, float]]) -> None:
    """
    entries: list of (basename.png, x_px, y_px)
    """
    lines = []
    lines.append("# Define the number of dimensions we are working on")
    lines.append("dim = 2")
    lines.append("")
    lines.append("# Define the image coordinates")
    for name, x, y in entries:
        lines.append(f"{name}; ; ({x:.3f}, {y:.3f})")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# --------------------------------- CLI ---------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert SEM NPZ to PNG tiles and build TileConfiguration.txt")
    p.add_argument("--sem-folder", required=True, type=Path, help="Folder with *_sem.npz files.")
    p.add_argument("--metadata-folder", required=True, type=Path, help="Folder with *_metadata.txt files.")
    p.add_argument("--outdir", type=Path, default=Path("processeddata"), help="Output base folder (default: processeddata).")
    p.add_argument("--png-subdir", type=str, default="sem-images-png", help="Subfolder under outdir for PNGs.")
    p.add_argument("--tileconfig-name", type=str, default="TileConfiguration.txt", help="Output filename for TileConfiguration.")
    p.add_argument("--glob", type=str, default="*_sem.npz", help="Glob for SEM files.")
    # normalization
    p.add_argument("--norm", choices=["auto","absolute","absolute16","fixed"], default="auto", help="Intensity normalization.")
    p.add_argument("--vmin", type=float, default=None, help="Used if --norm fixed.")
    p.add_argument("--vmax", type=float, default=None, help="Used if --norm fixed.")
    # stage → pixel mapping
    p.add_argument("--um-per-px", type=float, default=0.5490099, help="Micrometers per pixel for SEM tiles (default: 0.5490099).")
    p.add_argument("--invert-x", action="store_true", help="Invert stage X before normalization (mirror around Y).")
    p.add_argument("--invert-y", action="store_true", help="Invert stage Y before normalization (mirror around X).")
    args = p.parse_args(argv)

    sem_dir: Path = args.sem_folder
    meta_dir: Path = args.metadata_folder
    outdir: Path = args.outdir
    png_dir: Path = outdir / args.png_subdir
    png_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sem_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No SEM files found in {sem_dir} with pattern {args.glob}", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(files)} SEM file(s). Norm={args.norm}, µm/px={args.um_per_px}")

    # First pass: write PNGs, collect stage positions if available
    records = []  # (basename.png, x_um, y_um) — raw stage values in µm
    for i, sem_path in enumerate(files, 1):
        base_full = sem_path.stem
        base = base_full[:-4] if base_full.endswith("_sem") else base_full

        # PNG conversion
        try:
            arr = load_sem_npz(sem_path)
        except Exception as e:
            print(f"[ERROR] {sem_path.name}: load failed → {e}", file=sys.stderr)
            continue

        img8 = to_uint8(arr, method=args.norm, vmin=args.vmin, vmax=args.vmax)
        out_png = png_dir / f"{base}_sem.png"
        Image.fromarray(img8, mode="L").save(out_png)
        print(f"[{i}/{len(files)}] wrote {out_png.name} ({arr.shape[1]}×{arr.shape[0]})")

        # metadata → stage positions
        meta_path = meta_dir / f"{base}_metadata.txt"
        if not meta_path.exists():
            # try same folder as SEM file
            alt = sem_path.with_name(f"{base}_metadata.txt")
            meta_path = alt if alt.exists() else meta_path

        meta = parse_metadata_txt(meta_path) if meta_path.exists() else {}
        if not meta:
            print(f"    [WARN] metadata missing/empty for {base} — will exclude from TileConfiguration")
            continue

        x_mm = meta_num(meta, "Stage Position/X")
        y_mm = meta_num(meta, "Stage Position/Y")
        if x_mm is None or y_mm is None:
            print(f"    [WARN] stage positions not found for {base} — will exclude from TileConfiguration")
            continue

        x_um, y_um = x_mm * 1000.0, y_mm * 1000.0
        records.append((out_png.name, x_um, y_um))

    if not records:
        print("[WARN] No records with valid stage positions; TileConfiguration.txt will not be written.", file=sys.stderr)
        return 0

    # Second pass: apply flips, normalize to (0,0), convert µm → px
    names, xs_um, ys_um = zip(*records)
    xs = np.array(xs_um, dtype=float)
    ys = np.array(ys_um, dtype=float)

    # Apply flips (important: BEFORE zero-shift)
    if args.invert_x:
        xs = -xs
    if args.invert_y:
        ys = -ys

    # Normalize origin to top-left in the chosen frame
    xs -= xs.min()
    ys -= ys.min()

    xs_px = xs / args.um_per_px
    ys_px = ys / args.um_per_px

    # Build entries; sort by y then x (purely cosmetic)
    order = np.lexsort((xs_px, ys_px))
    entries = [(names[i], float(xs_px[i]), float(ys_px[i])) for i in order]

    tile_path = outdir / args.tileconfig_name
    write_tileconfig(tile_path, entries)
    print(f"[OK] Wrote {tile_path} with {len(entries)} entries")
    print("     Open this from the same folder in Fiji (no Invert X/Y needed).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
