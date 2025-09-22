#!/usr/bin/env python3
"""
Stitch NPZ-backed SEM tiles (from H5 pipeline) into pyramidal BigTIFF(s).

Inputs in <h5data_folder>:
  - summary_table.csv (must include: X_rel_um, Y_rel_um, npz_path; Z_layer optional)
  - *_sem.npz (key 'sem_data' -> 2D raw array; dtype often int16/uint16/uint8)

Examples:
  python stitch_h5data.py ./h5data --scale 0.5 --norm auto --clip-percent 1.0
  python stitch_h5data.py ./h5data --norm fixed --lo 4000 --hi 18000
  python stitch_h5data.py ./h5data --px_x_um 0.632324 --px_y_um 0.632324
"""

import os, math, argparse, shutil, random
from pathlib import Path
import numpy as np
import pandas as pd
import pyvips
from tqdm import tqdm

# --- libvips runtime config (env var can be overridden by CLI) ---------------
os.environ.setdefault("VIPS_CONCURRENCY", "2")
os.environ.setdefault("VIPS_PROGRESS", "1")

# ----------------------------- CSV helpers -----------------------------------

def must_have_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = ["X_rel_um", "Y_rel_um", "npz_path"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"summary_table.csv missing required column(s): {miss}")
    if "Z_layer" not in df.columns:
        df = df.copy()
        df["Z_layer"] = 0
    return df

def infer_px_um_from_df(df: pd.DataFrame, arg_px_x_um, arg_px_y_um) -> tuple[float, float]:
    """
    Resolve µm/px. Priority:
      1) --px_x_um/--px_y_um (if both provided),
      2) median of df['px_x_um'] and df['px_y_um'] if present/non-null,
      3) exit with a helpful message.
    """
    if arg_px_x_um is not None and arg_px_y_um is not None:
        return float(arg_px_x_um), float(arg_px_y_um)

    if {"px_x_um", "px_y_um"}.issubset(df.columns):
        px_x = df["px_x_um"].dropna()
        px_y = df["px_y_um"].dropna()
        if not px_x.empty and not px_y.empty:
            return float(px_x.median()), float(px_y.median())

    print(
        "px_x_um and/or px_y_um information not found in the metadata. "
        "If you know this information, please pass --px_x_um and --px_y_um."
    )
    raise SystemExit(1)

# ----------------------------- image helpers ---------------------------------

_mask_cache: dict[tuple[int,int,int], pyvips.Image] = {}

def feather_mask_vips(w: int, h: int, fpx: int) -> pyvips.Image:
    """Hann ramp 0..1 float where interior=1, edges→0 over fpx px."""
    if fpx <= 0:
        return pyvips.Image.black(w, h).bandor(1).cast("float")
    key = (w, h, fpx)
    wm = _mask_cache.get(key)
    if wm is not None:
        return wm
    xy = pyvips.Image.xyz(w, h)
    X, Y = xy[0].cast("float"), xy[1].cast("float")
    dx = (X < (w - 1 - X)).ifthenelse(X, (w - 1 - X))
    dy = (Y < (h - 1 - Y)).ifthenelse(Y, (h - 1 - Y))
    d  = (dx < dy).ifthenelse(dx, dy)
    r  = d / float(fpx)
    r1 = (r > 1).ifthenelse(1, r)
    wm = (0.5 * (1 - (r1 * math.pi).cos())).cast("float")
    _mask_cache[key] = wm
    return wm

def load_npz_sem(path: Path) -> np.ndarray | None:
    """Load raw 2D sem_data from *_sem.npz; return None on any failure."""
    try:
        with np.load(path) as z:
            a = z["sem_data"]
        if a.ndim == 3:
            a = a[..., 0]
        return np.asarray(a)
    except Exception:
        return None

def numpy_to_vips_gray(a: np.ndarray) -> tuple[pyvips.Image, str]:
    """
    Convert 2D numpy -> vips single-band image; return (image, fmt_str).
    Keeps 16-bit signed/unsigned when present; else 8-bit.
    """
    if a.dtype == np.uint16:
        fmt = "ushort"
    elif a.dtype == np.int16:
        fmt = "short"
    elif a.dtype == np.uint8:
        fmt = "uchar"
    else:
        # sensible fallback: prefer 16-bit signed for other integer types, else 8-bit
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.int16, copy=False)
            fmt = "short"
        else:
            a = a.astype(np.uint8, copy=False)
            fmt = "uchar"

    h, w = a.shape
    im = pyvips.Image.new_from_memory(np.ascontiguousarray(a).data, w, h, 1, fmt)
    return im.copy(interpretation="b-w"), fmt

def replicate(img: pyvips.Image, n: int) -> pyvips.Image:
    return pyvips.Image.bandjoin([img] * n)

def fmt_output_range(fmt: str) -> tuple[float, float]:
    """Return (min,max) representable values for a vips format."""
    if fmt == "uchar":
        return 0.0, 255.0
    if fmt == "ushort":
        return 0.0, 65535.0
    if fmt == "short":
        return -32768.0, 32767.0
    return 0.0, 255.0

# ---- percentile helpers (no libvips percentiles; sample NPZs directly) ------

def sample_npz_percentiles(dfz: pd.DataFrame, p_lo: float, p_hi: float,
                           samples_per_tile: int = 20000,
                           rng_seed: int = 0) -> tuple[float, float]:
    """
    Draw a small random sample per tile and compute percentiles in numpy.
    Robust and independent of libvips build.
    """
    rng = np.random.default_rng(rng_seed)
    acc = []
    for pth in dfz["npz_path"]:
        a = load_npz_sem(Path(pth))
        if a is None or a.size == 0:
            continue
        a = a.ravel()
        if samples_per_tile > 0 and a.size > samples_per_tile:
            idx = rng.choice(a.size, samples_per_tile, replace=False)
            a = a[idx]
        acc.append(a.astype(np.float64, copy=False))
    if not acc:
        return 0.0, 1.0
    x = np.concatenate(acc)
    lo, hi = np.nanpercentile(x, [p_lo, 100.0 - p_lo]) if p_hi is None else np.nanpercentile(x, [p_lo, p_hi])
    return float(lo), float(hi)

# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stitch H5/NPZ SEM tiles to pyramidal BigTIFF per Z layer.")
    ap.add_argument("h5data_folder", help="Folder containing summary_table.csv and *_sem.npz files")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <h5data_folder>/stitched)")
    ap.add_argument("--scale", type=float, default=0.5, help="Master scale factor (1.0 = full-res)")
    ap.add_argument("--feather-px", type=int, default=0, help="Feather width in pixels (0 = no feather)")
    ap.add_argument("--background", choices=["white","black"], default="white")
    ap.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--preview-max", type=int, default=1600)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vips-workers", type=int, default=2)
    ap.add_argument("--channel", choices=["gray", "rgb"], default="gray",
                    help="Export single-band grayscale (default) or replicate to RGB")
    # pixel-size overrides (µm/px)
    ap.add_argument("--px_x_um", type=float, default=None, help="Override pixel size along X in µm/px")
    ap.add_argument("--px_y_um", type=float, default=None, help="Override pixel size along Y in µm/px")
    # normalization options (like ndtiff)
    ap.add_argument("--norm", choices=["none","auto","fixed"], default="none",
                    help="Intensity normalization: none (raw), auto (percentiles), fixed (lo/hi)")
    ap.add_argument("--clip-percent", type=float, default=1.0,
                    help="For --norm auto: clip percent at both ends (e.g., 1.0 → p1/p99)")
    ap.add_argument("--lo", type=float, default=None, help="For --norm fixed: low clip (raw units)")
    ap.add_argument("--hi", type=float, default=None, help="For --norm fixed: high clip (raw units)")
    # percentile sampling controls
    ap.add_argument("--auto-samples-per-tile", type=int, default=20000,
                    help="When --norm auto, how many pixels to sample per tile (0=all)")
    ap.add_argument("--auto-rng-seed", type=int, default=0, help="RNG seed for auto sampling")

    args = ap.parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(args.vips_workers)

    in_dir = Path(args.h5data_folder)
    if not in_dir.is_dir():
        raise SystemExit(f"Folder not found: {in_dir}")

    csv_path = in_dir / "summary_table.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing file: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "stitched")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = in_dir.name

    df = must_have_columns(pd.read_csv(csv_path))
    # resolve px sizes (µm/px)
    px_x_um, px_y_um = infer_px_um_from_df(df, args.px_x_um, args.px_y_um)

    # determine Z layers
    try:
        S = int(df["Z_layer"].max()) + 1
    except Exception:
        S = 1

    # canvas in native px, then apply scale
    x_max_um = float((df["X_rel_um"] + df.get("TileWidth_um", 0)).max())
    y_max_um = float((df["Y_rel_um"] + df.get("TileHeight_um", 0)).max())
    Wn = int(math.ceil(x_max_um / px_x_um))
    Hn = int(math.ceil(y_max_um / px_y_um))
    SCALE = float(args.scale)
    W = max(1, int(round(Wn * SCALE)))
    H = max(1, int(round(Hn * SCALE)))
    print(f"Canvas (native): {Wn}×{Hn}  -> scale={SCALE}  =>  {W}×{H}")

    # space info
    try:
        tmp_dir = Path(os.environ.get("VIPS_TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP") or ".")
        _, _, out_free = shutil.disk_usage(out_dir)
        _, _, tmp_free = shutil.disk_usage(tmp_dir)
        print(f"[info] Free space -> out: {out_free/1024**3:.1f} GB, tmp: {tmp_free/1024**3:.1f} GB (TMP={tmp_dir})")
    except Exception:
        pass

    for z in range(S):
        dfz = df[df["Z_layer"] == z]
        if dfz.empty:
            print(f"[warn] no rows for Z={z}; skipping.")
            continue

        # probe dtype + global min/max from NPZ tiles (for this Z)
        probed_fmt = None
        global_min = np.inf
        global_max = -np.inf
        for pth in dfz["npz_path"]:
            a = load_npz_sem(Path(pth))
            if a is None or a.size == 0:
                continue
            if probed_fmt is None:
                _, probed_fmt = numpy_to_vips_gray(a)
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
            if amin < global_min: global_min = amin
            if amax > global_max: global_max = amax

        if not np.isfinite(global_min) or not np.isfinite(global_max):
            print(f"[warn] Z{z}: could not read any NPZ tiles; skipping.")
            continue

        # bit-depth guess for display
        bit_guess = "8-bit" if (global_min >= 0 and global_max <= 255) else "16-bit"
        print(f"\nZ{z} NPZ stats -> min={global_min:.3g}, max={global_max:.3g}  |  dtype={probed_fmt}  (looks {bit_guess})")

        # background in *image fmt* range
        if probed_fmt == "ushort":
            bgval_dtype = 65535 if args.background == "white" else 0
        elif probed_fmt == "short":
            bgval_dtype = 32767 if args.background == "white" else -32768
        else:  # uchar
            bgval_dtype = 255 if args.background == "white" else 0

        tiff_path = out_dir / f"{base_name}_Z{z}.tif"
        prev_path = out_dir / f"{base_name}_Z{z}_preview.png"

        if tiff_path.exists() and not args.overwrite:
            print(f"[skip] {tiff_path} exists.")
            # still produce a preview if needed
            if not prev_path.exists():
                try:
                    im = pyvips.Image.new_from_file(str(tiff_path), access="sequential")
                    s = min(1.0, args.preview_max / max(im.width, im.height))
                    prv = im.resize(s) if s < 1.0 else im
                    prv8 = _preview_from_vips(prv, None, None)  # 8-bit display
                    prv8.pngsave(str(prev_path), compression=6)
                    print(f"✅ Preview (from existing): {prev_path}")
                except Exception as e:
                    print(f"[warn] cannot create preview from existing TIFF: {e}")
            continue

        # compose mosaic (weighted if feather > 0)
        use_rgb = (args.channel == "rgb")
        if args.feather-px if False else False:  # satisfy linter (will be overridden below)
            pass
        if args.feather_px > 0:
            if use_rgb:
                acc  = pyvips.Image.black(W, H, bands=3).cast("float")
                wacc = pyvips.Image.black(W, H).cast("float")
            else:
                acc  = pyvips.Image.black(W, H, bands=1).cast("float")
                wacc = pyvips.Image.black(W, H).cast("float")

            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                x0 = int(round((r.X_rel_um / px_x_um) * SCALE))
                y0 = int(round((r.Y_rel_um / px_y_um) * SCALE))

                a = load_npz_sem(Path(r.npz_path))
                if a is None:
                    continue
                tile_v, tfmt = numpy_to_vips_gray(a)
                if SCALE != 1.0:
                    tile_v = tile_v.resize(SCALE, kernel="linear")

                wm = feather_mask_vips(tile_v.width, tile_v.height, int(args.feather_px))
                if use_rgb:
                    tile_v = replicate(tile_v, 3)
                    acc    = acc.insert(tile_v * replicate(wm, 3), x0, y0)
                else:
                    acc    = acc.insert(tile_v * wm, x0, y0)
                wacc = wacc.insert(wm, x0, y0)

            eps = 1e-6
            if use_rgb:
                w3   = replicate(wacc + eps, 3)
                mosaic = (acc / w3)
            else:
                mosaic = (acc / (wacc + eps))

            # holes to background
            if use_rgb:
                bg = (pyvips.Image.black(W, H, bands=3) + [bgval_dtype]*3).cast(probed_fmt)
                mosaic = (wacc > 0).ifthenelse(mosaic.cast(probed_fmt), bg)
            else:
                bg = (pyvips.Image.black(W, H) + bgval_dtype).cast(probed_fmt)
                mosaic = (wacc > 0).ifthenelse(mosaic.cast(probed_fmt), bg)

        else:
            # fast insert path
            if use_rgb:
                mosaic = (pyvips.Image.black(W, H, bands=3).cast(probed_fmt)
                          .copy(interpretation="srgb")) + [bgval_dtype]*3
            else:
                mosaic = (pyvips.Image.black(W, H).cast(probed_fmt)
                          .copy(interpretation="b-w")) + bgval_dtype

            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                x0 = int(round((r.X_rel_um / px_x_um) * SCALE))
                y0 = int(round((r.Y_rel_um / px_y_um) * SCALE))
                a  = load_npz_sem(Path(r.npz_path))
                if a is None:
                    continue
                tile_v, tfmt = numpy_to_vips_gray(a)
                if SCALE != 1.0:
                    tile_v = tile_v.resize(SCALE, kernel="linear")
                if use_rgb:
                    mosaic = mosaic.insert(replicate(tile_v, 3), x0, y0)
                else:
                    mosaic = mosaic.insert(tile_v, x0, y0)

        # ---------------- Normalization (optional) ----------------
        # Apply to the *mosaic* so the whole image uses a consistent mapping.
        lo_used = None
        hi_used = None

        if args.norm == "fixed":
            if args.lo is None or args.hi is None or args.hi <= args.lo:
                raise SystemExit("--norm fixed requires valid --lo <v> and --hi <v> (hi>lo).")
            lo_used, hi_used = float(args.lo), float(args.hi)

        elif args.norm == "auto":
            p = float(args.clip_percent)
            lo_used, hi_used = sample_npz_percentiles(
                dfz, p_lo=p, p_hi=100.0 - p, samples_per_tile=args.auto_samples_per_tile, rng_seed=args.auto_rng_seed
            )
            # sanity fallback
            if not np.isfinite(lo_used) or not np.isfinite(hi_used) or hi_used <= lo_used:
                lo_used, hi_used = global_min, global_max

        # apply mapping if requested
        if lo_used is not None and hi_used is not None:
            out_min, out_max = fmt_output_range(probed_fmt)
            scale  = (out_max - out_min) / max(hi_used - lo_used, 1e-6)
            mosaic = ((mosaic - lo_used) * scale + out_min).cast(probed_fmt)

        # save pyramidal BigTIFF
        try:
            mosaic = mosaic.copy(interpretation="srgb" if use_rgb else "b-w")
        except Exception:
            pass

        mosaic.tiffsave(
            str(tiff_path),
            tile=True,
            compression="lzw",
            bigtiff=True,
            pyramid=args.pyramid,
            predictor=True,
        )
        print(f"✅ BigTIFF written: {tiff_path}")

        # --- Preview PNG (always 8-bit for display) ---
        s = min(1.0, args.preview_max / max(W, H))
        prv = mosaic.resize(s) if s < 1.0 else mosaic
        # For preview, reuse chosen lo/hi if we normalized; else compute from NPZ samples
        if lo_used is None or hi_used is None:
            plo = min(1.0, args.clip_percent)
            lo_prev, hi_prev = sample_npz_percentiles(
                dfz, p_lo=plo, p_hi=100.0 - plo, samples_per_tile=min(5000, args.auto_samples_per_tile), rng_seed=args.auto_rng_seed
            )
        else:
            lo_prev, hi_prev = lo_used, hi_used

        prv8 = _preview_from_vips(prv, lo_prev, hi_prev)  # 8-bit display
        prv8.pngsave(str(prev_path), compression=6)
        print(f"✅ Preview PNG:   {prev_path}")

    print("All done.")

# ---------- preview helper (8-bit display mapping, robust) --------------------

def _preview_from_vips(vimg: pyvips.Image, lo: float | None, hi: float | None) -> pyvips.Image:
    """
    Return an 8-bit display image.
    If lo/hi provided, use them; else use vimg min/max.
    """
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        try:
            lo = float(vimg.min())
            hi = float(vimg.max())
        except Exception:
            lo, hi = 0.0, 1.0
    scaled = ((vimg - lo) * (255.0 / max(hi - lo, 1e-6))).cast("uchar")
    if scaled.bands == 1:
        return scaled.copy(interpretation="b-w")
    return scaled.copy(interpretation="srgb")

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
