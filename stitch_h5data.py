#!/usr/bin/env python3
"""
Stitch SEM NPZ tiles into a pyramidal BigTIFF + preview PNG.

Inputs
------
- A folder produced by the notebook, containing:
  - summary_table.csv   (one row per tile)
  - *_sem.npz           (NPZ files with 'sem_data' 2D arrays)

The BigTIFF it writes is designed to flow directly into dzi_from_bigtiff.py.

Example
-------
python3 stitch_h5data.py "/path/to/folder" \
  --scale 1.0 --norm global --clip-percent 0.05 --gamma 1.0 \
  --out-depth 8 --feather-px 40 --workers 12
"""

import os, math, argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd
import pyvips

# ---------- env before heavy vips work ----------
# (can be overridden by CLI)
os.environ.setdefault("VIPS_CONCURRENCY", "2")
os.environ.setdefault("VIPS_PROGRESS",   "1")

# ---------- CSV columns we need ----------
REQUIRED_COLS = [
    # geometry
    "X_rel_um", "Y_rel_um", "TileWidth_um", "TileHeight_um",
    "width_px", "height_px", "px_x_um", "px_y_um",
    # image path
    "npz_path",
]

OPTIONAL_COLS = [
    "invertx", "inverty",  # 0/1 flags (applied to stage coords before relative)
    # anything else is carried through but not required here
]

# ---------- small helpers ----------
def fmt_range(fmt: str) -> tuple[float, float]:
    if fmt == "uchar":
        return 0.0, 255.0
    if fmt == "ushort":
        return 0.0, 65535.0
    if fmt == "short":
        return -32768.0, 32767.0
    if fmt in ("float", "double"):
        return 0.0, 1.0
    # fallback
    return 0.0, 1.0

def depth_to_fmt(depth: int) -> tuple[str, float, float]:
    if depth == 8:
        return "uchar", 0.0, 255.0
    else:
        return "ushort", 0.0, 65535.0

def np_to_vips(arr: np.ndarray) -> pyvips.Image:
    """
    Create a vips Image from a 2D or 3D numpy array (H,W[,C]).
    The array must be C-contiguous; we ensure a copy if needed.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    height, width = arr.shape[:2]
    bands = 1 if arr.ndim == 2 else arr.shape[2]
    fmt_map = {
        np.uint8: "uchar",
        np.int16: "short",
        np.uint16: "ushort",
        np.float32: "float",
        np.float64: "double",
        np.int32: "int",
    }
    base = fmt_map.get(arr.dtype.type, None)
    if base is None:
        raise SystemExit(f"Unsupported numpy dtype: {arr.dtype}")

    # vips expects interleaved bands; memoryview is OK
    mem = arr.reshape(height * width * bands)
    vi = pyvips.Image.new_from_memory(
        mem.data, width, height, bands, base
    )
    return vi

def load_sem_npz(path: Path) -> np.ndarray:
    """
    Load NPZ produced by the convert step.
    Expects key 'sem_data' (2D image), dtype uint8/uint16/int16.
    """
    with np.load(path) as npz:
        if "sem_data" not in npz:
            # backward-compatible fallback: first array
            key = list(npz.files)[0]
            arr = npz[key]
        else:
            arr = npz["sem_data"]
    # normalize to 2D
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise SystemExit(f"{path}: expected 2D array, got shape {arr.shape}")
    return arr

def robust_percentiles_from_npz(rows: pd.DataFrame,
                                p_lo: float,
                                p_hi: float,
                                samples_per_tile: int,
                                seed: int) -> tuple[float, float]:
    """
    Sample pixels across many NPZ tiles to compute global percentiles.
    p_lo/p_hi in [0,100], samples_per_tile=0 -> take all pixels (slow).
    """
    assert 0 <= p_lo <= 100 and 0 <= p_hi <= 100
    rng = np.random.default_rng(seed)
    samples = []
    for _, r in rows.iterrows():
        try:
            a = load_sem_npz(Path(r["npz_path"])).astype(np.float64, copy=False)
            a = a.ravel()
            if samples_per_tile and a.size > samples_per_tile:
                idx = rng.choice(a.size, samples_per_tile, replace=False)
                a = a[idx]
            samples.append(a)
        except Exception:
            continue
    if not samples:
        raise SystemExit("Could not sample NPZs for global percentiles.")

    allv = np.concatenate(samples)
    lo = float(np.nanpercentile(allv, p_lo))
    hi = float(np.nanpercentile(allv, p_hi))
    return lo, hi

def make_feather_mask(w: int, h: int, fpx: int) -> pyvips.Image:
    """
    Create an alpha feather mask (uchar 0..255), cosine ramp of width fpx.
    """
    if fpx <= 0:
        return pyvips.Image.black(w, h) + 255

    # coordinate images
    x = pyvips.Image.xyz(w, h)[0]
    y = pyvips.Image.xyz(w, h)[1]

    # distances to each edge
    d_l = x
    d_r = (w - 1) - x
    d_t = y
    d_b = (h - 1) - y

    d = d_l.min(d_r).min(d_t).min(d_b).cast("float")

    # cosine ramp: d>=fpx => 1, d<=0 => 0
    # ramp = 0.5 * (1 - cos(pi * clamp(d/fpx, 0, 1)))
    ramp = (d / float(fpx)).max(0.0).min(1.0)
    ramp = ( (1.0 - (ramp * math.pi).cos()) * 0.5 )

    alpha = (ramp * 255.0).cast("uchar")
    return alpha

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Stitch NPZ SEM tiles into a pyramidal BigTIFF + preview."
    )
    ap.add_argument("folder", help="Folder containing summary_table.csv and NPZs")
    ap.add_argument("--summary", default="summary_table.csv",
                    help="CSV file name (default: summary_table.csv)")
    ap.add_argument("--out-dir", default="stitched",
                    help="Subfolder for outputs (default: 'stitched')")
    ap.add_argument("--z", type=int, default=0,
                    help="Z layer to stitch (default 0)")

    # Geometry / canvas
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Additional global scale (1.0 = native)")

    # Normalization
    ap.add_argument("--norm",
                    choices=["none", "absolute", "absolute16", "fixed", "global"],
                    default="global",
                    help="Normalization mode (default: global)")
    ap.add_argument("--clip-percent", type=float, default=1.0,
                    help="For --norm global: percentile per tail (default 1.0)")
    ap.add_argument("--lo", type=float, default=None,
                    help="For --norm fixed: low bound in raw DN")
    ap.add_argument("--hi", type=float, default=None,
                    help="For --norm fixed: high bound in raw DN")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Gamma after normalization (default 1.0)")

    # Output
    ap.add_argument("--out-depth", choices=[8, 16], type=int, default=8,
                    help="Bit depth of stitched BigTIFF (default 8)")
    ap.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True,
                    help="Write pyramidal BigTIFF (default true)")
    ap.add_argument("--preview-max", type=int, default=1600,
                    help="Max dimension for preview PNG (default 1600)")

    # Blending / placement
    ap.add_argument("--feather-px", type=int, default=0,
                    help="Feather width (px) for tile edges (default 0 = hard edges)")

    # Performance
    ap.add_argument("--workers", type=int, default=2,
                    help="libvips worker threads (best-effort)")
    ap.add_argument("--vips-w", type=int, default=256,
                    help="VIPS tile lines in buffer (default 256)")

    # Sampling for global stats
    ap.add_argument("--auto-samples-per-tile", type=int, default=5000,
                    help="Samples per tile for global percentiles (0=all)")
    ap.add_argument("--auto-rng-seed", type=int, default=0,
                    help="RNG seed for sampling")

    return ap.parse_args()

# ---------- normalization pipeline ----------
def apply_norm_np(arr: np.ndarray,
                  mode: str,
                  lo: float|None, hi: float|None,
                  global_lo: float|None, global_hi: float|None,
                  gamma: float,
                  out_depth: int) -> np.ndarray:
    """
    Normalize a 2D numpy array per settings, return uint8/uint16.
    """
    a = arr.astype(np.float32, copy=False)

    # choose lo/hi
    if mode == "none":
        a_min, a_max = float(np.min(a)), float(np.max(a))
        lo_used, hi_used = a_min, a_max
    elif mode == "absolute":
        # use dtype's full range
        if arr.dtype == np.int16:
            lo_used, hi_used = -32768.0, 32767.0
        elif arr.dtype == np.uint16:
            lo_used, hi_used = 0.0, 65535.0
        else:
            lo_used, hi_used = float(np.min(a)), float(np.max(a))
    elif mode == "absolute16":
        lo_used, hi_used = 0.0, 65535.0
    elif mode == "fixed":
        if lo is None or hi is None or hi <= lo:
            raise SystemExit("--norm fixed requires valid --lo < --hi")
        lo_used, hi_used = float(lo), float(hi)
    elif mode == "global":
        if global_lo is None or global_hi is None or global_hi <= global_lo:
            raise SystemExit("global lo/hi not available; recompute sampling.")
        lo_used, hi_used = float(global_lo), float(global_hi)
    else:
        raise SystemExit(f"Unknown --norm '{mode}'")

    # map to 0..1
    scale01 = 1.0 / max(hi_used - lo_used, 1e-6)
    n = (a - lo_used) * scale01
    n = np.clip(n, 0.0, 1.0)

    # gamma
    if gamma and gamma != 1.0:
        n = np.power(n, gamma, dtype=np.float32)

    # to depth
    if out_depth == 8:
        out = np.round(n * 255.0).astype(np.uint8)
    else:
        out = np.round(n * 65535.0).astype(np.uint16)

    return out

# ---------- main ----------
def main():
    args = parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(args.workers)
    pyvips.voperation.cache_set_max(0)  # avoid RAM spikes for large mosaics
    pyvips.vimage.Image.tilecache_set_max_mem(256 * 1024 * 1024)

    folder = Path(args.folder)
    summary_path = folder / args.summary
    out_dir = folder / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # output names
    zlab = f"Z{args.z}"
    tiff_path = out_dir / f"{folder.name}_{zlab}.tif"
    prev_path = out_dir / f"{folder.name}_{zlab}_preview.png"

    # load CSV
    df = pd.read_csv(summary_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {summary_path}: {missing}")

    # filter by z (if the notebook wrote Z_layer)
    if "Z_layer" in df.columns:
        df = df[df["Z_layer"] == args.z].copy().reset_index(drop=True)

    # sanity
    if len(df) == 0:
        raise SystemExit("No rows to stitch for the requested Z.")

    # invert flags (apply BEFORE relative coords in the notebook; here we just honor columns if present)
    invx = int(df.get("invertx", 0).iloc[0]) if "invertx" in df.columns else 0
    invy = int(df.get("inverty", 0).iloc[0]) if "inverty" in df.columns else 0

    # microns -> pixels scale using px size; assume uniform pixel size across set
    px_x_um = float(df["px_x_um"].dropna().iloc[0])
    px_y_um = float(df["px_y_um"].dropna().iloc[0])

    # per-tile pixel sizes
    w_px = df["width_px"].astype(int).to_numpy()
    h_px = df["height_px"].astype(int).to_numpy()

    # stage → pixel positions
    X_rel_um = df["X_rel_um"].astype(float).to_numpy()
    Y_rel_um = df["Y_rel_um"].astype(float).to_numpy()
    if invx:
        X_rel_um = (X_rel_um.max() - X_rel_um)
    if invy:
        Y_rel_um = (Y_rel_um.max() - Y_rel_um)

    x_px = (X_rel_um / px_x_um) * args.scale
    y_px = (Y_rel_um / px_y_um) * args.scale
    tw_px = w_px * args.scale
    th_px = h_px * args.scale

    # canvas size (ceil for safety)
    W = int(math.ceil(float((x_px + tw_px).max())))
    H = int(math.ceil(float((y_px + th_px).max())))

    # report
    print(f"Canvas (native): {W}x{H}  -> scale={args.scale}")
    free_out = shutil_disk_free(out_dir)
    free_tmp = shutil_disk_free(Path("."))  # crude
    print(f"[info] Free space -> out: {free_out:.1f} GB, tmp: {free_tmp:.1f} GB (TMP=.)")

    # global NPZ min/max (quick) for log, and optional percentiles
    raw_min = None
    raw_max = None
    try:
        mins = []
        maxs = []
        for p in df["npz_path"].head(16):  # cheap probe
            a = load_sem_npz(Path(p))
            mins.append(int(a.min()))
            maxs.append(int(a.max()))
        raw_min = min(mins) if mins else None
        raw_max = max(maxs) if maxs else None
    except Exception:
        pass

    dtype_hint = "short (looks 16-bit)" if any(
        load_sem_npz(Path(p)).dtype == np.int16 for p in df["npz_path"].head(1)
    ) else "uint8/uint16"
    if raw_min is not None:
        print(f"Z{args.z} NPZ stats -> min={raw_min}, max={raw_max}  |  dtype={dtype_hint}")

    # compute global lo/hi if requested
    glo = ghi = None
    if args.norm == "global":
        p = float(args.clip_percent)
        lo_p = max(0.0, p)
        hi_p = 100.0 - max(0.0, p)
        glo, ghi = robust_percentiles_from_npz(
            df, lo_p, hi_p, args.auto_samples_per_tile, args.auto_rng_seed
        )
        print(f"Global NPZ clipping @ {args.clip_percent}%: lo={int(round(glo))}, hi={int(round(ghi))}")

    # setup base canvas
    out_fmt, out_min, out_max = depth_to_fmt(args.out_depth)
    use_alpha = args.feather_px > 0
    if use_alpha:
        base = pyvips.Image.black(W, H, bands=2)  # gray + alpha
        base = base + [0, 0]  # ensure type is uchar/ushort? We'll cast on composite
    else:
        base = pyvips.Image.black(W, H)
    base = base.cast(out_fmt)
    if use_alpha:
        # alpha = 0 means transparent; fill 0
        base = base.bandjoin_const([0])

    # place tiles
    lines_in_buffer = max(64, int(args.vips_w))
    pyvips.vimage.Image.set_kill(False)

    # composition loop
    n = len(df)
    for i, row in df.iterrows():
        npz_path = Path(row["npz_path"])
        a = load_sem_npz(npz_path)

        # normalize to selected depth
        a_out = apply_norm_np(
            a,
            mode=args.norm,
            lo=args.lo, hi=args.hi,
            global_lo=glo, global_hi=ghi,
            gamma=args.gamma,
            out_depth=args.out_depth,
        )
        tile = np_to_vips(a_out)  # bands=1, format out_fmt
        tile = tile.copy(interpretation="b-w")

        # optional feather
        if use_alpha:
            fpx = int(args.feather_px)
            mask = make_feather_mask(tile.width, tile.height, fpx)  # uchar
            if out_fmt != "uchar":
                # upcast mask to output depth alpha (0..255 -> 0..65535)
                mask = (mask * (65535.0/255.0)).cast(out_fmt)
            tile_a = tile.bandjoin(mask)
            # composite 'over' at (x,y)
            xi = int(round((row["X_rel_um"] if not invx else X_rel_um[i]) / px_x_um * args.scale))
            yi = int(round((row["Y_rel_um"] if not invy else Y_rel_um[i]) / px_y_um * args.scale))
            base = base.composite2(tile_a, "over", x=xi, y=yi)
        else:
            # simple insert
            xi = int(round(x_px[i]))
            yi = int(round(y_px[i]))
            base = base.insert(tile, xi, yi)

        if (i + 1) % 128 == 0 or i + 1 == n:
            pct = int(round(100.0 * (i + 1) / n))
            print(f"Z{args.z} compose: {pct}%\r", end="", flush=True)
    print(f"Z{args.z} compose: 100%")

    mosaic = base

    # TIFF mapping report (we report the *intended* mapping from raw -> output range)
    # For per-tile normalization we print the global/fixed/absolute inputs we used.
    if args.norm == "global":
        lo_used, hi_used = glo, ghi
    elif args.norm == "fixed":
        lo_used, hi_used = args.lo, args.hi
    elif args.norm == "absolute16":
        lo_used, hi_used = 0, 65535
    elif args.norm == "absolute":
        # just log raw_min/max as hints
        lo_used, hi_used = raw_min, raw_max
    else:  # none
        lo_used = hi_used = None

    if lo_used is None or hi_used is None:
        try:
            lo_used = int(mosaic.min())
            hi_used = int(mosaic.max())
        except Exception:
            lo_used = hi_used = 0

    scale_factor = (out_max - out_min) / max(float(hi_used) - float(lo_used), 1e-6)
    print("TIFF mapping:",
          f"norm={args.norm}, clip%={args.clip_percent}, gamma={args.gamma},",
          f"raw_lo={int(lo_used)}, raw_hi={int(hi_used)},",
          f"out_fmt={out_fmt}, out_min={int(out_min)}, out_max={int(out_max)},",
          f"scale={scale_factor:.6f} DN_per_raw")

    # Write pyramidal BigTIFF — overwrite if exists
    if tiff_path.exists():
        tiff_path.unlink(missing_ok=True)

    mosaic = mosaic.copy(interpretation="b-w")
    if args.out_depth == 8:
        # For DZI pipelines, JPEG saves space and is fine visually; switch to LZW if you need lossless
        mosaic.tiffsave(str(tiff_path), tile=True, compression="jpeg", Q=90,
                        bigtiff=True, pyramid=args.pyramid,
                        tile_width=256, tile_height=256)
    else:
        mosaic.tiffsave(str(tiff_path), tile=True, compression="lzw",
                        bigtiff=True, pyramid=args.pyramid, predictor=True,
                        tile_width=256, tile_height=256)

    print(f"✅ BigTIFF written: {tiff_path}")

    # Preview PNG that uses the exact same numeric range as TIFF
    s = min(1.0, args.preview_max / max(W, H))
    prv = mosaic.resize(s) if s < 1.0 else mosaic
    # If 16-bit output, map to 0..255 for PNG consistently with TIFF range
    p_lo = out_min
    p_hi = out_max
    prv8 = ((prv - p_lo) * (255.0 / max(p_hi - p_lo, 1e-6))).cast("uchar")
    if prev_path.exists():
        prev_path.unlink(missing_ok=True)
    prv8.pngsave(str(prev_path), compression=6)

    print(f"✅ Preview PNG:   {prev_path}")
    print(f"    Preview mapping: scaled_from=[{int(p_lo)}, {int(p_hi)}] -> [0,255], "
          f"preview_size={prv.width}x{prv.height}")
    print("All done.")

def shutil_disk_free(path: Path) -> float:
    """Return free space in GB for display."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(path))
        return free / (1024**3)
    except Exception:
        return float('nan')

if __name__ == "__main__":
    main()
