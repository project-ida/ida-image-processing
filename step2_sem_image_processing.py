#!/usr/bin/env python3
"""
Step 2 — SEM image → tiles + Fiji TileConfiguration.txt

Adds:
- TileConfiguration.txt is written into the *tiles folder*
- --auto-um-per-px : compute µm/px from metadata (X/Y Step) + image size
- --out-format     : png8 or tiff16 (16-bit preserves dynamic range)
- --gamma          : gamma correction after normalization (gamma<1 brightens)
- --auto-clip-percent : (auto only) clip low/high percentiles to boost contrast
- Per-tile print of normalization values actually applied
- Corner sanity print (which files wind up at TL/TR/BL/BR)

Notes:
- Fiji: when you generate the layout with the flags here, leave
  “Invert X/Y coordinates” *unchecked* in the stitching dialog.
"""

from __future__ import annotations
import argparse, sys, re
from pathlib import Path
import numpy as np
from PIL import Image

# -------------------------- Metadata helpers --------------------------

def parse_metadata_txt(path: Path) -> dict[str, str]:
    """Accepts both 'key = value' and 'key: value' lines."""
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s or ":" in s:
            try:
                k, v = re.split(r"\s*[=:]\s*", s, maxsplit=1)
                meta[k.strip()] = v.strip()
            except Exception:
                continue
    return meta

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def meta_num(meta: dict[str, str], key_suffix: str, default: float | None = None) -> float | None:
    """Return first numeric value for a key that ends with key_suffix (case-insensitive)."""
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

def compute_lo_hi(arr: np.ndarray, method: str,
                  vmin: float | None, vmax: float | None,
                  auto_clip_percent: float) -> tuple[float, float]:
    """Decide normalization range (lo, hi)."""
    a = arr.astype(np.float64, copy=False)
    if method == "auto":
        if auto_clip_percent and auto_clip_percent > 0:
            lo, hi = np.nanpercentile(a, [auto_clip_percent, 100.0 - auto_clip_percent])
        else:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    elif method == "fixed":
        if vmin is None or vmax is None or vmax <= vmin:
            raise ValueError("--norm fixed requires valid --vmin < --vmax")
        lo, hi = float(vmin), float(vmax)
    elif method == "absolute16":
        lo, hi = 0.0, 65535.0
    elif method == "absolute":
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype); lo, hi = float(info.min), float(info.max)
        else:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    else:
        raise ValueError(f"Unknown --norm '{method}'")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return lo, hi

def scale_to_uint8(arr: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    """Scale arr in [lo,hi] → uint8 with gamma (gamma<1 brightens)."""
    a = arr.astype(np.float64, copy=False)
    a = (a - lo) / max(hi - lo, 1e-12)
    a = np.clip(a, 0.0, 1.0)
    if gamma and gamma != 1.0:
        a = np.power(a, gamma)
    return (a * 255.0 + 0.5).astype(np.uint8)

def scale_to_uint16(arr: np.ndarray, method: str, lo: float, hi: float, gamma: float) -> np.ndarray:
    """
    16-bit output:
    - absolute16: pass-throughish (clip 0..65535), ignores gamma/lo/hi
    - others    : scale to 0..65535, apply gamma
    """
    if method == "absolute16":
        a = np.clip(arr, 0.0, 65535.0)
        return a.astype(np.uint16)
    a = arr.astype(np.float64, copy=False)
    a = (a - lo) / max(hi - lo, 1e-12)
    a = np.clip(a, 0.0, 1.0)
    if gamma and gamma != 1.0:
        a = np.power(a, gamma)
    return (a * 65535.0 + 0.5).astype(np.uint16)

# ----------------------- TileConfiguration.txt -------------------------

def write_tileconfig(out_path: Path, entries: list[tuple[str, float, float]]) -> None:
    """entries: list of (basename, x_px, y_px)."""
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
    p = argparse.ArgumentParser(description="Convert SEM NPZ to tiles and build TileConfiguration.txt")
    p.add_argument("--sem-folder", required=True, type=Path, help="Folder with *_sem.npz files.")
    p.add_argument("--metadata-folder", required=True, type=Path, help="Folder with *_metadata.txt files.")
    p.add_argument("--outdir", type=Path, default=Path("processeddata"), help="Output base folder.")
    p.add_argument("--tiles-subdir", type=str, default="sem-images-png", help="Subfolder (under outdir) for tiles.")
    p.add_argument("--tileconfig-name", type=str, default="TileConfiguration.txt", help="Name for the Fiji layout file.")
    p.add_argument("--glob", type=str, default="*_sem.npz", help="Glob for SEM files.")
    # normalization
    p.add_argument("--out-format", choices=["png8","tiff16"], default="png8", help="Tile image format.")
    p.add_argument("--norm", choices=["auto","absolute","absolute16","fixed"], default="auto", help="Intensity normalization.")
    p.add_argument("--vmin", type=float, default=None, help="Used if --norm fixed.")
    p.add_argument("--vmax", type=float, default=None, help="Used if --norm fixed.")
    p.add_argument("--auto-clip-percent", type=float, default=0.0,
                   help="If --norm auto: clip low/high percentiles by this amount (e.g., 1.0).")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Gamma correction after normalization (gamma<1 brightens; >1 darkens).")
    # stage → pixel mapping
    p.add_argument("--um-per-px", type=float, default=0.5490099, help="µm per pixel (used unless --auto-um-per-px).")
    p.add_argument("--auto-um-per-px", action="store_true",
                   help="Derive µm/px from metadata X/Y Step and image size (median across tiles).")
    p.add_argument("--invert-x", action="store_true", help="Invert stage X before normalization (mirror around Y).")
    p.add_argument("--invert-y", action="store_true", help="Invert stage Y before normalization (mirror around X).")
    args = p.parse_args(argv)

    sem_dir: Path = args.sem_folder
    meta_dir: Path = args.metadata_folder
    outdir: Path = args.outdir
    tiles_dir: Path = outdir / args.tiles_subdir
    tiles_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sem_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No SEM files found in {sem_dir} with pattern {args.glob}", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(files)} SEM file(s). out={args.out_format}, norm={args.norm}, gamma={args.gamma}, auto-clip={args.auto_clip_percent}")

    records = []      # (basename, x_um, y_um)
    umppx_x_vals = [] # from X Step / width
    umppx_y_vals = [] # from Y Step / height

    for i, sem_path in enumerate(files, 1):
        base_full = sem_path.stem
        base = base_full[:-4] if base_full.endswith("_sem") else base_full

        # load image
        try:
            arr = np.asarray(load_sem_npz(sem_path))
        except Exception as e:
            print(f"[ERROR] {sem_path.name}: load failed → {e}", file=sys.stderr)
            continue

        H, W = arr.shape

        # compute lo/hi for normalization & export
        lo, hi = compute_lo_hi(arr, args.norm, args.vmin, args.vmax, args.auto_clip_percent)

        out_name = f"{base}_sem." + ("png" if args.out_format == "png8" else "tif")
        out_path = tiles_dir / out_name
        if args.out_format == "png8":
            img8 = scale_to_uint8(arr, lo, hi, args.gamma)
            Image.fromarray(img8, mode="L").save(out_path)
        else:  # tiff16
            img16 = scale_to_uint16(arr, args.norm, lo, hi, args.gamma)
            Image.fromarray(img16, mode="I;16").save(out_path)

        print(f"[{i}/{len(files)}] {out_path.name} ({W}×{H})  "
              f"norm={args.norm} lo={lo:.3g} hi={hi:.3g} gamma={args.gamma}")

        # metadata
        meta_path = meta_dir / f"{base}_metadata.txt"
        if not meta_path.exists():
            alt = sem_path.with_name(f"{base}_metadata.txt")
            if alt.exists():
                meta_path = alt
        meta = parse_metadata_txt(meta_path) if meta_path.exists() else {}
        if not meta:
            print(f"    [WARN] metadata missing/empty for {base} — skip in TileConfiguration")
            continue

        x_mm = meta_num(meta, "Stage Position/X")
        y_mm = meta_num(meta, "Stage Position/Y")
        if x_mm is None or y_mm is None:
            print(f"    [WARN] stage positions not found for {base} — skip in TileConfiguration")
            continue

        # gather µm/px candidates if available
        x_step_mm = meta_num(meta, "X Step")
        y_step_mm = meta_num(meta, "Y Step")
        if x_step_mm:
            umppx_x_vals.append((x_step_mm * 1000.0) / W)
        if y_step_mm:
            umppx_y_vals.append((y_step_mm * 1000.0) / H)

        records.append((out_name, x_mm * 1000.0, y_mm * 1000.0))

    if not records:
        print("[WARN] No records with valid stage positions; TileConfiguration.txt will not be written.", file=sys.stderr)
        return 0

    # choose µm/px
    um_per_px = args.um_per_px
    if args.auto_um_per_px:
        candidates = []
        if umppx_x_vals: candidates.append(np.median(umppx_x_vals))
        if umppx_y_vals: candidates.append(np.median(umppx_y_vals))
        if candidates:
            um_per_px = float(np.median(candidates))
            print(f"[INFO] --auto-um-per-px → using {um_per_px:.6f} µm/px "
                  f"(from {len(umppx_x_vals)} X and {len(umppx_y_vals)} Y samples)")
        else:
            print(f"[WARN] --auto-um-per-px requested but no X/Y Step found; using --um-per-px={um_per_px}", file=sys.stderr)
    else:
        print(f"[INFO] Using manual --um-per-px = {um_per_px}")

    # coords → pixels
    names, xs_um, ys_um = zip(*records)
    xs = np.array(xs_um, dtype=float)
    ys = np.array(ys_um, dtype=float)
    if args.invert_x: xs = -xs
    if args.invert_y: ys = -ys
    xs -= xs.min()
    ys -= ys.min()
    xs_px = xs / um_per_px
    ys_px = ys / um_per_px

    # sort only for readability
    order = np.lexsort((xs_px, ys_px))
    entries = [(names[i], float(xs_px[i]), float(ys_px[i])) for i in order]

    # Write TileConfiguration into the *tiles* folder
    tile_path = (tiles_dir / args.tileconfig_name)
    write_tileconfig(tile_path, entries)
    print(f"[OK] Wrote {tile_path} with {len(entries)} entries")
    print("     Open this file in Fiji from the same folder; leave 'Invert X/Y' unchecked.")

    # Corner sanity
    tl = int(np.argmin(xs + ys))
    tr = int(np.argmin((-xs) + ys))
    bl = int(np.argmin(xs + (-ys)))
    br = int(np.argmax(xs + ys))
    print("\n[Corner sanity]")
    print(f"  top-left     : {names[tl]}")
    print(f"  top-right    : {names[tr]}")
    print(f"  bottom-left  : {names[bl]}")
    print(f"  bottom-right : {names[br]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
