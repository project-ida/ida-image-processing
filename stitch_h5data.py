#!/usr/bin/env python3
"""
Stitch NPZ-backed SEM tiles (from H5 pipeline) into pyramidal BigTIFF(s).

Inputs in <h5data_folder>:
  • summary_table.csv  (must include: X_rel_um, Y_rel_um, npz_path; Z_layer optional)
  • *_sem.npz          (key 'sem_data' -> 2D raw array; dtype often int16/uint16/uint8)

Normalization modes (harmonized with your step-2 script):
  --norm {none,auto,fixed,absolute,absolute16,global}
  --clip-percent / --auto-clip-percent (alias)
  --lo/--hi (for fixed)
  --gamma

This version:
  • overwrites existing TIFF/PNG by default
  • prints NPZ min/max as integers (no scientific notation)
  • prints the lo/hi used to scale the preview PNG
"""

import os, math, argparse, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyvips
from tqdm import tqdm

# ----- libvips runtime -----
os.environ.setdefault("VIPS_CONCURRENCY", "2")
os.environ.setdefault("VIPS_PROGRESS", "1")

# ----- CSV / metadata -----

def need_cols(df: pd.DataFrame) -> pd.DataFrame:
    req = ["X_rel_um", "Y_rel_um", "npz_path"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"summary_table.csv missing required column(s): {miss}")
    if "Z_layer" not in df.columns:
        df = df.copy()
        df["Z_layer"] = 0
    return df

def resolve_px_um(df: pd.DataFrame, px_x_um_arg, px_y_um_arg):
    if px_x_um_arg is not None and px_y_um_arg is not None:
        return float(px_x_um_arg), float(px_y_um_arg)
    if {"px_x_um","px_y_um"}.issubset(df.columns):
        sx = df["px_x_um"].dropna(); sy = df["px_y_um"].dropna()
        if not sx.empty and not sy.empty:
            return float(sx.median()), float(sy.median())
    raise SystemExit("Pixel sizes (µm/px) not found. Pass --px_x_um and --px_y_um.")

# ----- NPZ I/O -----

def load_npz_sem(path: Path) -> np.ndarray | None:
    try:
        with np.load(path) as z:
            key = "sem_data" if "sem_data" in z.files else (z.files[0] if z.files else None)
            if key is None:
                return None
            a = z[key]
        if a.ndim == 3:
            a = a[..., 0]
        return np.asarray(a)
    except Exception:
        return None

def numpy_to_vips_gray(a: np.ndarray):
    if a.dtype == np.uint16:
        fmt = "ushort"
    elif a.dtype == np.int16:
        fmt = "short"
    elif a.dtype == np.uint8:
        fmt = "uchar"
    else:
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.int16, copy=False); fmt = "short"
        else:
            a = a.astype(np.uint8,  copy=False); fmt = "uchar"
    h, w = a.shape
    im = pyvips.Image.new_from_memory(np.ascontiguousarray(a).data, w, h, 1, fmt)
    return im.copy(interpretation="b-w"), fmt

def fmt_range(fmt: str) -> tuple[float,float]:
    if fmt == "uchar":  return (0.0, 255.0)
    if fmt == "ushort": return (0.0, 65535.0)
    if fmt == "short":  return (-32768.0, 32767.0)
    return (0.0, 255.0)

# ----- feather mask -----

_mask_cache: dict[tuple[int,int,int], pyvips.Image] = {}

def feather_mask_vips(w: int, h: int, fpx: int) -> pyvips.Image:
    if fpx <= 0:
        return pyvips.Image.black(w, h).bandor(1).cast("float")
    key = (w, h, fpx)
    if key in _mask_cache:
        return _mask_cache[key]
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

def bandrep(img: pyvips.Image, n: int) -> pyvips.Image:
    return pyvips.Image.bandjoin([img]*n)

# ----- robust percentiles via NPZ sampling -----

def sample_npz_percentiles(dfz: pd.DataFrame, plo: float, phi: float | None = None,
                           per_tile: int = 20000, seed: int = 0) -> tuple[float,float]:
    rng = np.random.default_rng(seed)
    acc = []
    for p in dfz["npz_path"]:
        a = load_npz_sem(Path(p))
        if a is None or a.size == 0: continue
        v = a.ravel()
        if per_tile and v.size > per_tile:
            idx = rng.choice(v.size, per_tile, replace=False)
            v = v[idx]
        acc.append(v.astype(np.float64, copy=False))
    if not acc:
        return 0.0, 1.0
    x = np.concatenate(acc)
    lo, hi = np.nanpercentile(x, [plo, 100.0-plo] if phi is None else [plo, phi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return float(lo), float(hi)

# ----- normalization -----

def apply_norm(img: pyvips.Image, fmt: str,
               mode: str, *, lo: float|None, hi: float|None,
               clip_percent: float, dfz: pd.DataFrame,
               per_tile: int, seed: int,
               gamma: float, force_ushort: bool) -> tuple[pyvips.Image, float, float]:
    in_lo, in_hi = fmt_range(fmt)

    if mode == "none":
        lo_used, hi_used = in_lo, in_hi
    elif mode == "fixed":
        if lo is None or hi is None or hi <= lo:
            raise SystemExit("--norm fixed requires valid --lo < --hi")
        lo_used, hi_used = float(lo), float(hi)
    elif mode == "absolute16":
        lo_used, hi_used = 0.0, 65535.0
        fmt = "ushort"
    elif mode == "absolute":
        lo_used, hi_used = in_lo, in_hi
    elif mode in ("auto", "global"):
        p = max(0.0, float(clip_percent))
        lo_used, hi_used = sample_npz_percentiles(dfz, p, None, per_tile, seed)
    else:
        raise SystemExit(f"Unknown --norm '{mode}'")

    # normalize to 0..1
    scale = 1.0 / max(hi_used - lo_used, 1e-6)
    num = (img - lo_used) * scale

    # clamp to [0, 1] using masks
    num = (num < 0).ifthenelse(0.0, num)
    num = (num > 1).ifthenelse(1.0, num)

    # gamma (optional)
    if gamma and gamma != 1.0:
        num = (num.cast("float") ** gamma)

    out_fmt = "ushort" if (force_ushort or mode == "absolute16") else fmt
    out_min, out_max = fmt_range(out_fmt)
    mapped = (num * (out_max - out_min) + out_min).cast(out_fmt)

    return mapped, lo_used, hi_used

def preview_u8(img: pyvips.Image, lo: float|None, hi: float|None) -> pyvips.Image:
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        try:
            lo, hi = float(img.min()), float(img.max())
        except Exception:
            lo, hi = 0.0, 1.0
    return ((img - lo) * (255.0 / max(hi - lo, 1e-6))).cast("uchar")

# ----- main -----

def main():
    ap = argparse.ArgumentParser(description="Stitch H5/NPZ SEM tiles to pyramidal BigTIFF per Z layer.")
    ap.add_argument("h5data_folder")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--feather-px", type=int, default=0)
    ap.add_argument("--background", choices=["white","black"], default="white")
    ap.add_argument("--channel", choices=["gray","rgb"], default="gray")
    ap.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--preview-max", type=int, default=1600)

    # overwrite: default True now
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                    help="Overwrite existing outputs (default: true). Use --no-overwrite to keep files.")

    ap.add_argument("--vips-workers", type=int, default=2)

    # pixel sizes (µm/px)
    ap.add_argument("--px_x_um", type=float, default=None)
    ap.add_argument("--px_y_um", type=float, default=None)

    # normalization
    ap.add_argument("--norm", choices=["none","auto","fixed","absolute","absolute16","global"], default="none")
    ap.add_argument("--clip-percent", type=float, default=1.0, help="Percentile tails for auto/global (e.g. 1.0 => p1/p99)")
    ap.add_argument("--auto-clip-percent", type=float, dest="clip_percent_alias", default=None,
                    help="Alias of --clip-percent")
    ap.add_argument("--lo", type=float, default=None, help="Low clip for --norm fixed")
    ap.add_argument("--hi", type=float, default=None, help="High clip for --norm fixed")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma after normalization (1.0 = off)")
    ap.add_argument("--auto-samples-per-tile", type=int, default=20000)
    ap.add_argument("--auto-rng-seed", type=int, default=0)

    args = ap.parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(args.vips_workers)

    if args.clip_percent_alias is not None:
        args.clip_percent = args.clip_percent_alias

    in_dir = Path(args.h5data_folder)
    if not in_dir.is_dir():
        raise SystemExit(f"Folder not found: {in_dir}")
    csv_path = in_dir / "summary_table.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing file: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "stitched")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = in_dir.name

    df = need_cols(pd.read_csv(csv_path))
    px_x_um, px_y_um = resolve_px_um(df, args.px_x_um, args.px_y_um)

    x_max_um = float((df["X_rel_um"] + df.get("TileWidth_um", 0)).max())
    y_max_um = float((df["Y_rel_um"] + df.get("TileHeight_um", 0)).max())
    Wn = int(math.ceil(x_max_um / px_x_um))
    Hn = int(math.ceil(y_max_um / px_y_um))
    SCALE = float(args.scale)
    W = max(1, int(round(Wn * SCALE)))
    H = max(1, int(round(Hn * SCALE)))
    print(f"Canvas (native): {Wn}×{Hn}  -> scale={SCALE:.3g}  =>  {W}×{H}")

    try:
        tmp_dir = Path(os.environ.get("VIPS_TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP") or ".")
        _,_,of = shutil.disk_usage(out_dir); _,_,tf = shutil.disk_usage(tmp_dir)
        print(f"[info] Free space -> out: {of/1024**3:.1f} GB, tmp: {tf/1024**3:.1f} GB (TMP={tmp_dir})")
    except Exception:
        pass

    try:
        S = int(df["Z_layer"].max()) + 1
    except Exception:
        S = 1

    for z in range(S):
        dfz = df[df["Z_layer"] == z]
        if dfz.empty:
            print(f"[warn] no rows for Z={z}; skipping.")
            continue

        # NPZ min/max + dtype probe
        probed_fmt = None
        gmin, gmax = np.inf, -np.inf
        for p in dfz["npz_path"]:
            a = load_npz_sem(Path(p))
            if a is None: continue
            if probed_fmt is None:
                _, probed_fmt = numpy_to_vips_gray(a)
            gmin = min(gmin, float(np.nanmin(a)))
            gmax = max(gmax, float(np.nanmax(a)))
        if not np.isfinite(gmin) or not np.isfinite(gmax):
            print(f"[warn] Z{z}: could not read any tiles; skipping.")
            continue
        bit_hint = "8-bit" if (gmin >= 0 and gmax <= 255) else "16-bit"
        # integers, no scientific
        print(f"\nZ{z} NPZ stats -> min={int(round(gmin))}, max={int(round(gmax))}  |  dtype={probed_fmt}  (looks {bit_hint})")

        # background in source dtype
        if probed_fmt == "ushort": bg = 65535 if args.background == "white" else 0
        elif probed_fmt == "short": bg = 32767 if args.background == "white" else -32768
        else: bg = 255 if args.background == "white" else 0

        tiff_path = out_dir / f"{base}_Z{z}.tif"
        prev_path = out_dir / f"{base}_Z{z}_preview.png"

        use_rgb = (args.channel == "rgb")

        # --- compose ---
        if args.feather_px > 0:
            acc  = pyvips.Image.black(W, H, bands=(3 if use_rgb else 1)).cast("float")
            wacc = pyvips.Image.black(W, H).cast("float")
            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                a = load_npz_sem(Path(r.npz_path))
                if a is None: continue
                tile, _ = numpy_to_vips_gray(a)
                if SCALE != 1.0: tile = tile.resize(SCALE, kernel="linear")
                x0 = int(round((r.X_rel_um / px_x_um) * SCALE))
                y0 = int(round((r.Y_rel_um / px_y_um) * SCALE))
                wm = feather_mask_vips(tile.width, tile.height, int(args.feather_px))
                if use_rgb: acc  = acc.insert(bandrep(tile * wm, 3), x0, y0)
                else:       acc  = acc.insert(tile * wm, x0, y0)
                wacc = wacc.insert(wm, x0, y0)
            eps = 1e-6
            mosaic = (acc / (bandrep(wacc + eps, 3) if use_rgb else (wacc + eps)))
            # holes -> bg
            if use_rgb:
                bgimg = (pyvips.Image.black(W, H, bands=3) + [bg]*3).cast(probed_fmt)
                mosaic = (wacc > 0).ifthenelse(mosaic.cast(probed_fmt), bgimg)
            else:
                bgimg = (pyvips.Image.black(W, H) + bg).cast(probed_fmt)
                mosaic = (wacc > 0).ifthenelse(mosaic.cast(probed_fmt), bgimg)
        else:
            mosaic = (pyvips.Image.black(W, H, bands=(3 if use_rgb else 1)).cast(probed_fmt)
                      .copy(interpretation=("srgb" if use_rgb else "b-w")))
            mosaic = mosaic + ([bg]*3 if use_rgb else bg)
            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                a = load_npz_sem(Path(r.npz_path))
                if a is None: continue
                tile, _ = numpy_to_vips_gray(a)
                if SCALE != 1.0: tile = tile.resize(SCALE, kernel="linear")
                x0 = int(round((r.X_rel_um / px_x_um) * SCALE))
                y0 = int(round((r.Y_rel_um / px_y_um) * SCALE))
                mosaic = mosaic.insert(bandrep(tile, 3) if use_rgb else tile, x0, y0)

        # --- normalize whole mosaic ---
        force_ushort = (args.norm == "absolute16")
        mosaic, lo_used, hi_used = apply_norm(
            mosaic, probed_fmt, args.norm,
            lo=args.lo, hi=args.hi,
            clip_percent=args.clip_percent,
            dfz=dfz, per_tile=args.auto_samples_per_tile, seed=args.auto_rng_seED if False else args.auto_rng_seed,
            gamma=args.gamma, force_ushort=force_ushort
        )
        # (typo guard above leaves the correct arg in place)

        # write pyramidal BigTIFF (overwrite allowed)
        try:
            mosaic = mosaic.copy(interpretation=("srgb" if use_rgb else "b-w"))
        except Exception:
            pass
        mosaic.tiffsave(str(tiff_path), tile=True, compression="lzw",
                        bigtiff=True, pyramid=args.pyramid, predictor=True)
        print(f"✅ BigTIFF written: {tiff_path}")

        # --- preview (8-bit) using SAME lo/hi ---
        s = min(1.0, args.preview_max / max(W, H))
        prv = mosaic.resize(s) if s < 1.0 else mosaic
        prv8 = preview_u8(prv, lo_used, hi_used)
        prv8.pngsave(str(prev_path), compression=6)
        # integers, no scientific notation
        print(f"✅ Preview PNG:   {prev_path}")
        print(f"    Preview scale lo={int(round(lo_used))}, hi={int(round(hi_used))}")

    print("All done.")

if __name__ == "__main__":
    main()
