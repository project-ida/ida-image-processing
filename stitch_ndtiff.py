#!/usr/bin/env python3
import os, math, argparse
from pathlib import Path

# enable libvips progress + set workers BEFORE importing pyvips
os.environ.setdefault("VIPS_CONCURRENCY", "2")
os.environ.setdefault("VIPS_PROGRESS",   "1")

import numpy as np
import pandas as pd
import pyvips
from tqdm import tqdm
from ndstorage import Dataset


# ---------- helpers ----------
def must_have_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = ["X_rel_um", "Y_rel_um"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"summarytable.csv missing required columns: {miss}")
    if "Z_layer" not in df.columns:
        df = df.copy()
        df["Z_layer"] = 0
    return df

def parse_px_um_from_affine(summary) -> tuple[float,float]:
    # summary['UserData']['AffineTransform']['scalar'] is like "0.2_0.0_0.0_-0.2"
    affine = summary["UserData"]["AffineTransform"]["scalar"]
    ax, _bx, _by, dy = map(float, affine.split("_"))
    return abs(ax), abs(dy)  # µm/px in X and Y

def load_tile_pos_z(images, pos_index: int, z_layer: int) -> np.ndarray:
    a = images[pos_index, 0, 0, z_layer]
    try: a = a.compute()
    except Exception: pass
    a = np.asarray(a)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.shape[-1] == 4:
        a = a[..., :3]
    return a

def tile_to_vips(a: np.ndarray):
    if a.dtype == np.uint16:
        fmt, bg = "ushort", 65535
    else:
        a = a.astype(np.uint8, copy=False)
        fmt, bg = "uchar", 255
    h, w, c = a.shape
    im = pyvips.Image.new_from_memory(np.ascontiguousarray(a).data, w, h, c, fmt)
    return im.copy(interpretation="srgb"), fmt, bg

def replicate(img: pyvips.Image, n: int) -> pyvips.Image:
    return pyvips.Image.bandjoin([img] * n)

_mask_cache: dict[tuple[int,int,int], pyvips.Image] = {}

def feather_mask_vips(w: int, h: int, fpx: int) -> pyvips.Image:
    """Hann ramp 0..1 (float) where interior=1, edges→0 over fpx px."""
    if fpx <= 0:
        return pyvips.Image.black(w, h).bandor(1).cast("float")  # ones
    key = (w, h, fpx)
    wm = _mask_cache.get(key)
    if wm is not None:
        return wm
    xy = pyvips.Image.xyz(w, h)
    X, Y = xy[0].cast("float"), xy[1].cast("float")
    # distance to nearest vertical/horizontal edge (no min() op available, use conditionals)
    dx = (X < (w - 1 - X)).ifthenelse(X, (w - 1 - X))
    dy = (Y < (h - 1 - Y)).ifthenelse(Y, (h - 1 - Y))
    d  = (dx < dy).ifthenelse(dx, dy)
    r  = d / float(fpx)
    r1 = (r > 1).ifthenelse(1, r)  # clamp to [0,1]
    wm = (0.5 * (1 - (r1 * math.pi).cos())).cast("float")  # 0..1
    _mask_cache[key] = wm
    return wm


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Stitch NDTIFF tiles to pyramidal BigTIFF per Z layer (optional feather).")
    ap.add_argument("ndtiff_folder", help="Path to the NDTIFF dataset directory")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: ndtiff_folder)")
    ap.add_argument("--scale", type=float, default=0.5, help="Master scale factor (1.0 = full-res)")
    ap.add_argument("--feather-px", type=int, default=0, help="Feather width in pixels (0 = no feather)")
    ap.add_argument("--background", choices=["white","black"], default="white")
    ap.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--preview-max", type=int, default=1600)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vips-workers", type=int, default=2)
    args = ap.parse_args()

    os.environ["VIPS_CONCURRENCY"] = str(args.vips_workers)

    ndpath = Path(args.ndtiff_folder)
    csv_path = ndpath / "summarytable.csv"
    if not ndpath.is_dir():
        raise SystemExit(f"Folder not found: {ndpath}")
    if not csv_path.exists():
        raise SystemExit(f"Missing file: {csv_path}")

    df = must_have_columns(pd.read_csv(csv_path))
    ds = Dataset(str(ndpath))
    images = ds.as_array()
    summary = ds.summary_metadata

    px_x_um, px_y_um = parse_px_um_from_affine(summary)
    um_to_px_x = lambda um: um / px_x_um
    um_to_px_y = lambda um: um / px_y_um

    # Z layers (prefer summary)
    try:
        S = int(summary.get("Slices") or (images.shape[3] if images.ndim >= 4 else 1))
    except Exception:
        S = int(df["Z_layer"].max()) + 1

    # Canvas (native), then scale
    x_max_um = float((df["X_rel_um"] + df.get("TileWidth_um", 0)).max())
    y_max_um = float((df["Y_rel_um"] + df.get("TileHeight_um", 0)).max())
    Wn = int(math.ceil(um_to_px_x(x_max_um)))
    Hn = int(math.ceil(um_to_px_y(y_max_um)))
    SCALE = float(args.scale)
    W = max(1, int(round(Wn * SCALE)))
    H = max(1, int(round(Hn * SCALE)))
    print(f"Canvas (native): {Wn}×{Hn}  -> scale={SCALE}  =>  {W}×{H}")

    out_dir = Path(args.out_dir) if args.out_dir else ndpath
    out_dir.mkdir(parents=True, exist_ok=True)

    bgval = 255 if args.background.lower() == "white" else 0
    bg_rgb = [bgval, bgval, bgval]

    # Optional: cap VIPS cache a bit (keeps memory predictable)
    try:
        pyvips.cache_set_max_mem(1_000_000_000)  # ~1 GB
    except Exception:
        pass

    for z in range(S):
        dfz = df[df["Z_layer"] == z]
        if dfz.empty:
            print(f"[warn] no rows for Z={z}; skipping.")
            continue

        tiff_path = out_dir / f"mosaic_Z{z}.tif"
        prev_path = out_dir / f"mosaic_Z{z}_preview.png"

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

        # probe first tile for bit depth
        first = dfz.iloc[0]
        pos0 = int(first["PosIndex"]) if "PosIndex" in dfz.columns else (int(first["Index"]) // S)
        a0 = load_tile_pos_z(images, pos0, z)
        _, fmt, _ = tile_to_vips(a0)

        print(f"\n=== Z{z}: composing {len(dfz)} tiles on {W}×{H} (feather={args.feather_px}px) ===")

        if args.feather_px > 0:
            # weighted accumulation (float), then normalize
            acc  = pyvips.Image.black(W, H, bands=3).cast("float")
            wacc = pyvips.Image.black(W, H).cast("float")
            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                pos = int(getattr(r, "PosIndex", int(r.Index) // S))
                x0  = int(round(um_to_px_x(float(r.X_rel_um)) * SCALE))
                y0  = int(round(um_to_px_y(float(r.Y_rel_um)) * SCALE))

                a = load_tile_pos_z(images, pos, z)
                t, _fmt, _ = tile_to_vips(a)
                if SCALE != 1.0:
                    t = t.resize(SCALE, kernel="linear")
                t = t.cast("float")

                wm  = feather_mask_vips(t.width, t.height, int(args.feather_px))
                wm3 = replicate(wm, 3)

                acc  = acc.insert(t * wm3, x0, y0)
                wacc = wacc.insert(wm,      x0, y0)

            eps  = 1e-6
            w3   = replicate(wacc + eps, 3)
            outf = (acc / w3).cast(fmt)

            # fill empty areas with background
            cond = (wacc > 0)
            bgim = (pyvips.Image.black(W, H, bands=3) + bg_rgb).cast(fmt)
            mosaic = cond.ifthenelse(outf, bgim)

        else:
            # simple fast insert
            mosaic = (pyvips.Image.black(W, H, bands=3).cast(fmt)
                      .copy(interpretation="srgb")) + bg_rgb
            for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
                pos = int(getattr(r, "PosIndex", int(r.Index) // S))
                x0  = int(round(um_to_px_x(float(r.X_rel_um)) * SCALE))
                y0  = int(round(um_to_px_y(float(r.Y_rel_um)) * SCALE))

                a = load_tile_pos_z(images, pos, z)
                t, _fmt, _ = tile_to_vips(a)
                if SCALE != 1.0:
                    t = t.resize(SCALE, kernel="linear")
                mosaic = mosaic.insert(t, x0, y0)

        # save pyramidal BigTIFF (VIPS_PROGRESS prints progress)
        mosaic.tiffsave(
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
        prv = mosaic.resize(s) if s < 1.0 else mosaic
        prv.pngsave(str(prev_path), compression=6)
        print(f"✅ Preview PNG:   {prev_path}")

    print("All done.")

if __name__ == "__main__":
    main()
