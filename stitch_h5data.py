#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stitch SEM tiles from NPZ (h5->npz) into a single 16-bit pyramidal BigTIFF
plus an 8-bit preview PNG.

Usage (example):
  python stitch_h5data.py "/path/to/pythondata" \
      --scale 1.0 --norm global --clip-percent 1.0 --gamma 1.0 --workers 12 --overwrite

Notes
- BigTIFF is saved as 16-bit (ushort). Values are not 8-bit normalized.
- Preview PNG (8-bit) uses the same window (lo/hi) printed at the end.
- If you always run DZI on the BigTIFF, the DZI tool will coerce to 8-bit.
  Keeping the stitched TIFF 16-bit preserves dynamic range upstream.
"""

import os, sys, csv, math, time, argparse
from pathlib import Path

import numpy as np
import pyvips
from tqdm.auto import tqdm

# ---------- helpers ----------

def sample_npz_percentiles(npz_paths, p, sample_per_tile=5000, seed=0):
    """Return (lo, hi) percentiles across *all* NPZ tiles."""
    rng = np.random.default_rng(seed)
    samples = []
    for pth in npz_paths:
        try:
            a = np.load(pth)["sem_data"]      # shape (H, W), dtype typically uint16/int16-ish
            a = a.astype(np.int32, copy=False).ravel()
            if sample_per_tile and a.size > sample_per_tile:
                idx = rng.choice(a.size, sample_per_tile, replace=False)
                a = a[idx]
            samples.append(a)
        except Exception:
            continue
    if not samples:
        raise RuntimeError("No NPZ samples collected for percentile estimation.")
    allx = np.concatenate(samples)
    lo, hi = np.percentile(allx, [p, 100.0 - p])
    return float(lo), float(hi)

def read_summary_csv(csv_path: Path):
    """
    Read summary_table.csv -> list of dict rows.
    Required columns:
      npz_path,width_px,height_px,px_x_um,px_y_um,X_rel_um,Y_rel_um
    Optional: invertx,inverty,global_lo,global_hi
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise SystemExit(f"[error] No rows in {csv_path}")
    need = ["npz_path","width_px","height_px","px_x_um","px_y_um","X_rel_um","Y_rel_um"]
    for k in need:
        if k not in rows[0]:
            raise SystemExit(f"[error] '{k}' missing in {csv_path}")
    return rows

def parse_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def parse_int(x, default=None):
    try:
        return int(round(float(x)))
    except Exception:
        return default

def np_to_vips_u16(arr: np.ndarray) -> pyvips.Image:
    """
    Convert a 2D numpy array to a pyvips ushort image.
    - If arr is uint16: pass-through
    - If arr is int16 but all >=0: cast to uint16
    - If arr is uint8: up-convert (value * 257)
    - Else: clip to [0, 65535] and cast to uint16
    """
    a = arr
    if a.dtype == np.uint16:
        a16 = a
    elif a.dtype == np.int16:
        if a.min() >= 0:
            a16 = a.astype(np.uint16, copy=False)
        else:
            # shift negative to zero
            a16 = np.clip(a, 0, 65535).astype(np.uint16, copy=False)
    elif a.dtype == np.uint8:
        a16 = (a.astype(np.uint16) * 257)
    else:
        a16 = np.clip(a.astype(np.float64), 0, 65535).astype(np.uint16)

    h, w = a16.shape
    mem = a16.tobytes()
    img = pyvips.Image.new_from_memory(mem, w, h, 1, format="ushort")
    # match grayscale interpretation
    return img.copy(interpretation="b-w")

def build_canvas_u16(width: int, height: int) -> pyvips.Image:
    """Create an empty 16-bit grayscale canvas."""
    # Make a black uchar, then cast to ushort to get a zeroed 16-bit canvas.
    base = pyvips.Image.black(width, height, bands=1).cast("ushort")
    return base.copy(interpretation="b-w")

def insert_tile(canvas: pyvips.Image, tile: pyvips.Image, x: int, y: int) -> pyvips.Image:
    """Insert tile at (x,y)."""
    return canvas.insert(tile, x, y, expand=True)

def save_bigtiff_u16(img: pyvips.Image, out_path: Path, tile=128, workers=2, overwrite=False):
    """Save 16-bit pyramidal BigTIFF (deflate)."""
    os.environ["VIPS_CONCURRENCY"] = str(int(workers))
    tiff_opts = dict(
        compression="deflate",
        tile=True, tile_width=tile, tile_height=tile,
        pyramid=True,
        bigtiff=True,
        # 'Q' ignored for deflate; strip tags to reduce size
        strip=True
    )
    if (not overwrite) and out_path.exists():
        print(f"[skip] {out_path} exists (use --overwrite to replace)")
        return
    img.tiffsave(str(out_path), **tiff_opts)

def percentiles_from_samples(samples: np.ndarray, lo_p: float, hi_p: float):
    lo = float(np.percentile(samples, lo_p))
    hi = float(np.percentile(samples, hi_p))
    return lo, hi

def scale_to_u8(arr16: np.ndarray, lo: float, hi: float, gamma: float = 1.0) -> np.ndarray:
    """
    Clip ushort array to [lo, hi], normalize to 0..1, then apply gamma *darkening*
    for gamma>1 (standard convention). Finally map to uint8.
    """
    a = arr16.astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo)
    if gamma and abs(gamma - 1.0) > 1e-6:
        a = np.power(a, float(gamma))   # <-- was 1.0/gamma
    a = (a * 255.0 + 0.5).astype(np.uint8)
    return a

def save_preview_png(img16: pyvips.Image, out_png: Path, lo: float, hi: float, gamma: float = 1.0,
                     target_max_side: int = 1600, overwrite: bool = True):
    """
    Downsample and write an 8-bit PNG preview from a 16-bit vips Image using lo/hi.
    """
    if (not overwrite) and out_png.exists():
        print(f"[skip] {out_png} exists")
        return

    # Pull to numpy once (downscaling at the very end for simpler, exact control)
    arr = np.ndarray(
        buffer=img16.write_to_memory(),
        dtype=np.uint16,
        shape=(img16.height, img16.width, img16.bands),
    )[:, :, 0]  # gray

    arr8 = scale_to_u8(arr, lo, hi, gamma=gamma)

    # resize if necessary (keep aspect)
    h, w = arr8.shape
    scale = 1.0
    if max(w, h) > target_max_side:
        scale = target_max_side / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        import PIL.Image as pil
        arr8 = np.array(pil.fromarray(arr8, mode="L").resize((new_w, new_h), pil.BILINEAR))

    # write PNG via pyvips for speed
    out = pyvips.Image.new_from_memory(arr8.tobytes(), arr8.shape[1], arr8.shape[0], 1, "uchar")
    out = out.copy(interpretation="b-w")
    out.pngsave(str(out_png), compression=3, strip=True)

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Stitch NPZ SEM tiles into a 16-bit pyramidal BigTIFF + 8-bit preview.")
    p.add_argument("folder", help="Folder that contains summary_table.csv and NPZ files")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor for physical -> pixel (default 1.0)")
    p.add_argument("--tile", type=int, default=128, help="TIFF tile size (default 128)")
    p.add_argument("--workers", type=int, default=2, help="libvips worker threads (best-effort)")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                   help="Overwrite existing outputs (default true)")
    # normalization for PREVIEW ONLY (TIFF stays 16-bit)
    p.add_argument("--norm", choices=["none", "absolute", "global", "scan", "auto"], default="global",
                   help="Preview windowing strategy (TIFF is always 16-bit raw)")
    p.add_argument("--lo", type=float, default=None, help="Absolute lo (only with --norm absolute)")
    p.add_argument("--hi", type=float, default=None, help="Absolute hi (only with --norm absolute)")
    p.add_argument("--clip-percent", type=float, default=1.0,
                   help="Percentile clip used by scan/auto/global (default 1.0)")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma for preview (default 1.0)")
    p.add_argument("--preview-side", type=int, default=1600, help="Preview longest side in px (default 1600)")
    p.add_argument("--out-dir", default=None, help="Output dir for stitched files (default: <folder>/stitched)")
    return p.parse_args()

# ---------- main ----------

def main():
    args = parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(int(args.workers))
    folder = Path(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else (folder / "stitched")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    csv_path = folder / "summary_table.csv"
    rows = read_summary_csv(csv_path)

    # after you loaded summary_table.csv into df (or however you build the list)
    npz_paths = [
        r["npz_path"]
        for r in rows
        if r.get("npz_path") and os.path.isfile(r["npz_path"])
    ]
    if not npz_paths:
        raise SystemExit("No NPZ files found via 'npz_path' column.")
  
    # Apply invert flags if present, then compute pixel placements
    # Choose units: X_rel_um / px_x_um -> pixel offsets
    Xum = []
    Yum = []
    Wpx = []
    Hpx = []
    NPZ = []
    px_x_um = []
    px_y_um = []

    has_invx = "invertx" in rows[0]
    has_invy = "inverty" in rows[0]

    for r in rows:
        invx = parse_int(r.get("invertx", 0)) if has_invx else 0
        invy = parse_int(r.get("inverty", 0)) if has_invy else 0

        # relative um from CSV
        x_um = parse_float(r["X_rel_um"], 0.0)
        y_um = parse_float(r["Y_rel_um"], 0.0)
        if invx == 1:
            x_um = -x_um
        if invy == 1:
            y_um = -y_um

        Xum.append(x_um)
        Yum.append(y_um)
        Wpx.append(parse_int(r["width_px"], 0))
        Hpx.append(parse_int(r["height_px"], 0))
        NPZ.append(r["npz_path"])
        px_x_um.append(parse_float(r["px_x_um"], None))
        px_y_um.append(parse_float(r["px_y_um"], None))

    Xum = np.asarray(Xum, dtype=np.float64)
    Yum = np.asarray(Yum, dtype=np.float64)
    Wpx = np.asarray(Wpx, dtype=np.int32)
    Hpx = np.asarray(Hpx, dtype=np.int32)
    px_x_um = np.asarray(px_x_um, dtype=np.float64)
    px_y_um = np.asarray(px_y_um, dtype=np.float64)

    if np.any(~np.isfinite(px_x_um)) or np.any(~np.isfinite(px_y_um)):
        raise SystemExit("[error] px_x_um/px_y_um contain NaN/inf")

    # convert stage um -> pixel offsets (round to nearest int)
    x_off_px = np.round((Xum / px_x_um) * float(args.scale)).astype(np.int64)
    y_off_px = np.round((Yum / px_y_um) * float(args.scale)).astype(np.int64)
    # rebase so minimum is (0,0)
    x_off_px -= int(x_off_px.min())
    y_off_px -= int(y_off_px.min())

    # determine canvas size in px
    canvas_w = int((x_off_px + Wpx).max())
    canvas_h = int((y_off_px + Hpx).max())

    # quick dataset stats (for info & for preview when norm=none)
    g_min = None
    g_max = None

    print(f"Canvas (native): {canvas_w}x{canvas_h}   -> scale={args.scale}  =>  {canvas_w}x{canvas_h}")
    free_gb = None
    try:
        st = os.statvfs(str(out_dir))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        print(f"[info] Free space -> out: {free_gb:.1f} GB")
    except Exception:
        pass

    # Compose
    canvas = build_canvas_u16(canvas_w, canvas_h)
    t0 = time.time()
    bar = tqdm(total=len(NPZ), unit="tile", desc="Z0 compose")

    # also gather samples for auto/global preview window (fast reservoir sampling)
    samples = []
    rng = np.random.default_rng(0)
    SAMPLE_PER_TILE = 10000  # decent speed/robustness trade-off

    for i, npz_path in enumerate(NPZ):
        try:
            d = np.load(npz_path)
            # accept either 'sem_data' or first array
            if "sem_data" in d:
                a = d["sem_data"]
            else:
                key = next(iter(d.files))
                a = d[key]
        except Exception as e:
            bar.write(f"[warn] cannot read {npz_path}: {e}")
            bar.update(1)
            continue

        if a.ndim == 3:
            # (H,W,1) -> (H,W)
            if a.shape[-1] == 1:
                a = a[..., 0]
            else:
                a = a[..., 0]  # first band

        # global min/max
        amin = int(a.min())
        amax = int(a.max())
        g_min = amin if g_min is None else min(g_min, amin)
        g_max = amax if g_max is None else max(g_max, amax)

        # collect samples for preview window
        if a.size > SAMPLE_PER_TILE:
            idx = rng.choice(a.size, SAMPLE_PER_TILE, replace=False)
            samples.append(a.reshape(-1)[idx])
        else:
            samples.append(a.reshape(-1))

        # insert tile
        vx = np_to_vips_u16(a)
        canvas = insert_tile(canvas, vx, int(x_off_px[i]), int(y_off_px[i]))

        bar.update(1)

    bar.close()
    print(f"Z0 NPZ stats -> min={g_min}, max={g_max}  |  dtype≈short  (looks 16-bit)")
    # Save BigTIFF (always ushort pyramid)
    out_tif = out_dir / f"{folder.name.replace(' ', '_')}_Z0.tif"
    save_bigtiff_u16(canvas, out_tif, tile=args.tile, workers=args.workers, overwrite=args.overwrite)
    print(f"✅ BigTIFF written: {out_tif}")

    # ---------- preview normalization ----------
    lo_used = g_min
    hi_used = g_max

    def pick_lo_hi_from_norm(args, npz_paths, global_min, global_max):
        norm = (args.norm or "none").lower()
        if norm == "absolute":
            # user-supplied fixed window (ensure in the correct order)
            lo = float(args.lo) if args.lo is not None else global_min
            hi = float(args.hi) if args.hi is not None else global_max
            if hi < lo:
                lo, hi = hi, lo
            return lo, hi
    
        if norm == "global":
            # <-- this is where --clip-percent must take effect
            p = float(args.clip_percent or 1.0)
            return sample_npz_percentiles(
                npz_paths=npz_paths,
                p=p,
                sample_per_tile=getattr(args, "sample_per_tile", 5000),
                seed=getattr(args, "seed", 0),
            )
    
        if norm == "auto":
            # keep whatever “auto” logic you already had, or fallback to global_min/max
            return global_min, global_max
    
        # norm == "none" (or unknown): full raw range
        return global_min, global_max

    lo_used, hi_used = pick_lo_hi_from_norm(args, npz_paths, g_min, g_max)

    # preview
    out_png = out_dir / f"{folder.name.replace(' ', '_')}_Z0_preview.png"
    save_preview_png(canvas, out_png, lo=lo_used, hi=hi_used,
                     gamma=args.gamma, target_max_side=args.preview_side, overwrite=True)

    # Final report
    print(f"🛈 Preview scale lo={int(round(lo_used))}, hi={int(round(hi_used))}, gamma={args.gamma}")
    print("All done.")

if __name__ == "__main__":
    main()
