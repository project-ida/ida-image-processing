#!/usr/bin/env python3
"""
Stitch NPZ-backed SEM tiles (from H5 pipeline) to pyramidal BigTIFF per Z layer.

Expected inputs in <h5data_folder>:
  - summary_table.csv   (must include X_rel_um, Y_rel_um, and either TileWidth_um/TileHeight_um
                         or width_px/height_px + px_x_um/px_y_um, plus npz_path; Z_layer optional)
  - *_sem.npz           (key 'sem_data', raw 2D array per tile)

Example:
  python stitch_h5data.py /path/to/h5data --scale 0.5 --feather-px 30 --background white
"""

import os, math, argparse, shutil
from pathlib import Path

# configure libvips before importing pyvips
os.environ.setdefault("VIPS_CONCURRENCY", "2")
os.environ.setdefault("VIPS_PROGRESS",   "1")

import numpy as np
import pandas as pd
import pyvips
from tqdm import tqdm

# ---------------- helpers (ported/adapted from stitch_ndtiff.py) ----------------

def must_have_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = ["X_rel_um", "Y_rel_um", "npz_path"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"summary_table.csv missing required columns: {miss}")
    if "Z_layer" not in df.columns:
        df = df.copy()
        df["Z_layer"] = 0
    return df

def infer_px_um_from_df(df: pd.DataFrame, arg_px_x_um, arg_px_y_um) -> tuple[float, float]:
    """
    Resolve µm/px. Priority:
      1) --px_x_um/--px_y_um if both provided,
      2) df columns px_x_um / px_y_um (robust median over non-null),
      3) error with the same helpful message as ndtiff stitcher.
    """
    if arg_px_x_um is not None and arg_px_y_um is not None:
        return float(arg_px_x_um), float(arg_px_y_um)

    if {"px_x_um", "px_y_um"}.issubset(df.columns):
        px_x = df["px_x_um"].dropna()
        px_y = df["px_y_um"].dropna()
        if not px_x.empty and not px_y.empty:
            return float(px_x.median()), float(px_y.median())

    print("px_x_um and/or px_y_um information not found in the metadata. "
          "If you know this information, we can proceed by you entering it as parameters "
          "--px_x_um and --px_y_um")
    raise SystemExit(1)

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

def expected_tile_px_from_row(r, um_per_px_x, um_per_px_y):
    """
    Return (w,h) in px using TileWidth_um/TileHeight_um if present,
    else use width_px/height_px. None if insufficient info.
    """
    # Preferred path: um sizes + µm/px
    tw = getattr(r, "TileWidth_um", None)
    th = getattr(r, "TileHeight_um", None)
    if tw is not None and not pd.isna(tw) and th is not None and not pd.isna(th):
        try:
            w = int(round(float(tw) / um_per_px_x))
            h = int(round(float(th) / um_per_px_y))
            return (max(1, w), max(1, h))
        except Exception:
            pass

    # Fallback: explicit pixel sizes
    if hasattr(r, "width_px") and hasattr(r, "height_px"):
        try:
            w = int(r.width_px)
            h = int(r.height_px)
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            pass
    return None

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
    Keeps uint16 when present, else uint8.
    """
    if a.dtype == np.uint16:
        fmt = "ushort"
    else:
        if a.dtype != np.uint8:
            a = a.astype(np.uint8, copy=False)
        fmt = "uchar"
    h, w = a.shape
    im = pyvips.Image.new_from_memory(np.ascontiguousarray(a).data, w, h, 1, fmt)
    return im.copy(interpretation="b-w"), fmt

def replicate(img: pyvips.Image, n: int) -> pyvips.Image:
    return pyvips.Image.bandjoin([img] * n)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Stitch H5/NPZ SEM tiles to pyramidal BigTIFF per Z layer (optional feather).")
    ap.add_argument("h5data_folder", help="Folder containing summary_table.csv and *_sem.npz files")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <h5data_folder>/stitched)")
    ap.add_argument("--scale", type=float, default=0.5, help="Master scale factor (1.0 = full-res)")
    ap.add_argument("--feather-px", type=int, default=0, help="Feather width in pixels (0 = no feather)")
    ap.add_argument("--background", choices=["white","black"], default="white")
    ap.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--preview-max", type=int, default=1600)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vips-workers", type=int, default=2)
    # channel option for parity with ndtiff stitcher; SEM is 1-band, but allow 3-band export
    ap.add_argument("--channel", choices=["gray", "rgb"], default="gray",
                    help="Export single-band grayscale (default) or replicate to RGB")
    # optional pixel-size overrides (µm/px)
    ap.add_argument("--px_x_um", type=float, default=None, help="Override pixel size along X in µm/px")
    ap.add_argument("--px_y_um", type=float, default=None, help="Override pixel size along Y in µm/px")

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
    um_to_px_x = lambda um: um / px_x_um
    um_to_px_y = lambda um: um / px_y_um

    # determine number of Z layers
    try:
        S = int(df["Z_layer"].max()) + 1
    except Exception:
        S = 1

    # compute canvas (native), then scale
    x_max_um = float((df["X_rel_um"] + df.get("TileWidth_um", 0)).max())
    y_max_um = float((df["Y_rel_um"] + df.get("TileHeight_um", 0)).max())
    Wn = int(math.ceil(um_to_px_x(x_max_um)))
    Hn = int(math.ceil(um_to_px_y(y_max_um)))
    SCALE = float(args.scale)
    W = max(1, int(round(Wn * SCALE)))
    H = max(1, int(round(Hn * SCALE)))
    print(f"Canvas (native): {Wn}×{Hn}  -> scale={SCALE}  =>  {W}×{H}")

    bgval = 255 if args.background.lower() == "white" else 0
    if args.channel == "rgb":
        bg_rgb = [bgval, bgval, bgval]

    # Try to print disk space info
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

        tiff_path = out_dir / f"{base_name}_Z{z}.tif"
        prev_path = out_dir / f"{base_name}_Z{z}_preview.png"

        if tiff_path.exists() and not args.overwrite:
            print(f"[skip] {tiff_path} exists.")
            if not prev_path.exists():
                try:
                    im = pyvips.Image.new_from_file(str(tiff_path), access="sequential")
                    s = min(1.0, args.preview_max / max(im.width, im.height))
                    prv = im.resize(s) if s < 1.0 else im
                    prv.pngsave(str(prev_path), compression=6)
                    print(f"✅ Preview (from existing): {prev_path}")
                except Exception as e:
                    print(f"[warn] cannot create preview from existing TIFF: {e}")
            continue

        # Probe bit depth on first valid NPZ of this Z
        fmt = "uchar"  # default
        probe_done = False
        for r in dfz.itertuples(index=False):
            a = load_npz_sem(Path(getattr(r, "npz_path")))
            if a is None:
                continue
            fmt = "ushort" if a.dtype == np.uint16 else "uchar"
            probe_done = True
            break
        if not probe_done:
            print(f"[warn] Z{z}: no readable NPZ tiles found; skipping.")
            continue

        print(f"\n=== Z{z}: composing {len(dfz)} tiles on {W}×{H} "
              f"(feather={args.feather_px}px, channel={args.channel}) ===")

        bad_tiles = 0

        if args.feather_px > 0:
            # weighted accumulation
            if args.channel == "rgb":
                acc  = pyvips.Image.black(W, H, bands=3).cast("float")
                wacc = pyvips.Image.black(W, H).cast("float")
            else:
                acc  = pyvips.Image.black(W, H, bands=1).cast("float")
                wacc = pyvips.Image.black(W, H).cast("float")

            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                x0 = int(round(um_to_px_x(float(r.X_rel_um)) * SCALE))
                y0 = int(round(um_to_px_y(float(r.Y_rel_um)) * SCALE))

                a  = load_npz_sem(Path(getattr(r, "npz_path")))
                if a is None:
                    bad_tiles += 1
                    continue

                exp = expected_tile_px_from_row(r, px_x_um, px_y_um)
                if exp is not None:
                    aw, ah = exp
                    # tolerate small diffs (scanner rounding)
                    if not (abs(a.shape[1]-aw) <= 5 and abs(a.shape[0]-ah) <= 5):
                        # skip size mismatches to avoid mosaic corruption
                        bad_tiles += 1
                        continue

                t, tfmt = numpy_to_vips_gray(a)
                if SCALE != 1.0:
                    t = t.resize(SCALE, kernel="linear")

                wm = feather_mask_vips(t.width, t.height, int(args.feather_px))
                if args.channel == "rgb":
                    t_rgb = replicate(t, 3)
                    acc   = acc.insert(t_rgb * replicate(wm, 3), x0, y0)
                else:
                    acc   = acc.insert(t * wm, x0, y0)
                wacc = wacc.insert(wm, x0, y0)

            eps = 1e-6
            if args.channel == "rgb":
                w3   = replicate(wacc + eps, 3)
                outf = (acc / w3).cast(fmt)
                bgim = (pyvips.Image.black(W, H, bands=3) + [bgval]*3).cast(fmt)
                mosaic = (wacc > 0).ifthenelse(outf, bgim).copy(interpretation="srgb")
            else:
                outf = (acc / (wacc + eps)).cast(fmt)
                bgim = (pyvips.Image.black(W, H) + bgval).cast(fmt)
                mosaic = (wacc > 0).ifthenelse(outf, bgim).copy(interpretation="b-w")

        else:
            # fast insert
            if args.channel == "rgb":
                mosaic = (pyvips.Image.black(W, H, bands=3).cast(fmt)
                          .copy(interpretation="srgb")) + [bgval]*3
            else:
                mosaic = (pyvips.Image.black(W, H).cast(fmt)
                          .copy(interpretation="b-w")) + bgval

            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                x0 = int(round(um_to_px_x(float(r.X_rel_um)) * SCALE))
                y0 = int(round(um_to_px_y(float(r.Y_rel_um)) * SCALE))

                a  = load_npz_sem(Path(getattr(r, "npz_path")))
                if a is None:
                    bad_tiles += 1
                    continue

                exp = expected_tile_px_from_row(r, px_x_um, px_y_um)
                if exp is not None:
                    aw, ah = exp
                    if not (abs(a.shape[1]-aw) <= 5 and abs(a.shape[0]-ah) <= 5):
                        bad_tiles += 1
                        continue

                t, tfmt = numpy_to_vips_gray(a)
                if SCALE != 1.0:
                    t = t.resize(SCALE, kernel="linear")

                if args.channel == "rgb":
                    mosaic = mosaic.insert(replicate(t, 3), x0, y0)
                else:
                    mosaic = mosaic.insert(t, x0, y0)

        if bad_tiles:
            print(f"[warn] Z{z}: skipped {bad_tiles} tile(s) due to size/header/read issues.")

        # save pyramidal BigTIFF
        save_im = mosaic
        try:
            save_im = save_im.copy(interpretation="srgb" if args.channel == "rgb" else "b-w")
        except Exception:
            pass

        save_im.tiffsave(
            str(tiff_path),
            tile=True,
            compression="lzw",
            bigtiff=True,
            pyramid=args.pyramid,
            predictor=True,
        )
        print(f"✅ BigTIFF written: {tiff_path}")

        # preview
        s = min(1.0, args.preview_max / max(W, H))
        prv = save_im.resize(s) if s < 1.0 else save_im
        prv.pngsave(str(prev_path), compression=6)
        print(f"✅ Preview PNG:   {prev_path}")

    print("All done.")

if __name__ == "__main__":
    main()
