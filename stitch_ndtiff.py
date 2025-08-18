#!/usr/bin/env python3
import os, math, json, argparse
from pathlib import Path

# --- set VIPS env BEFORE importing pyvips ---
os.environ.setdefault("VIPS_CONCURRENCY", "2")  # best-effort on Windows
os.environ.setdefault("VIPS_PROGRESS", "1")     # shows % in terminal

import numpy as np
import pandas as pd
import pyvips
from tqdm import tqdm
from ndstorage import Dataset


# ---------------------- helpers ----------------------
def parse_affine_px(summary, px_x_um_override=None, px_y_um_override=None):
    """Get pixel size (µm/px) from metadata AffineTransform 'ax_bx_by_dy',
       or use overrides if provided. Returns (px_x_um, px_y_um)."""
    if px_x_um_override is not None and px_y_um_override is not None:
        return float(px_x_um_override), float(px_y_um_override)

    px_x_um = px_y_um = None
    aff = (summary.get("UserData", {}) or {}).get("AffineTransform", {})
    if isinstance(aff, dict):
        aff = aff.get("scalar", "")
    if isinstance(aff, str) and "_" in aff:
        try:
            ax, bx, by, dy = map(float, aff.split("_"))
            px_x_um = abs(ax) if ax else None
            px_y_um = abs(dy) if dy else None
        except Exception:
            pass
    # fallback if one side found
    if px_x_um is None and px_y_um is not None:
        px_x_um = px_y_um
    if px_y_um is None and px_x_um is not None:
        px_y_um = px_x_um
    # last-resort defaults
    if px_x_um is None or px_y_um is None:
        # conservative default; allow CLI overrides to refine
        px_x_um = px_x_um or 0.2
        px_y_um = px_y_um or 0.2
        print(f"[warn] Pixel size not in metadata; defaulting to {px_x_um} µm/px")
    return float(px_x_um), float(px_y_um)


def um_to_px_factory(px_x_um, px_y_um):
    return (lambda um: um / px_x_um), (lambda um: um / px_y_um)


def load_tile_pos_z(images, pos_index: int, z_layer: int):
    """Lazy-load one tile for (position, z). Keeps dtype (uint8/uint16), expands gray, drops alpha."""
    a = images[pos_index, 0, 0, z_layer]
    try:
        a = a.compute()
    except Exception:
        pass
    a = np.asarray(a)
    # drop leading singleton dims (rare)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    # expand gray / drop alpha
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.shape[-1] == 4:
        a = a[..., :3]
    return a


def tile_to_vips(a: np.ndarray):
    """Return (vips_image, fmt, bgval) where fmt in {'uchar','ushort'}."""
    if a.dtype == np.uint16:
        fmt, bgval = "ushort", 65535
    else:
        a = a.astype(np.uint8, copy=False)
        fmt, bgval = "uchar", 255
    h, w, c = a.shape
    img = pyvips.Image.new_from_memory(np.ascontiguousarray(a).data, w, h, c, fmt)
    return img.copy(interpretation="srgb"), fmt, bgval


def ensure_columns(df: pd.DataFrame):
    must_have = ["X_rel_um", "Y_rel_um"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise ValueError(f"summarytable.csv missing required columns: {missing}")
    # If Z_layer not present, assume 1 layer
    if "Z_layer" not in df.columns:
        df = df.copy()
        df["Z_layer"] = 0
    return df


# ---------------------- main ----------------------
def main():
    p = argparse.ArgumentParser(description="Stitch NDTIFF tiles into pyramidal BigTIFF per Z layer.")
    p.add_argument("ndtiff_folder", help="Path to NDTIFF dataset directory")
    p.add_argument("--out-dir", default=None, help="Output directory (default: ndtiff_folder)")
    p.add_argument("--scale", type=float, default=0.5, help="Master scale (e.g. 1.0 for full-res, 0.5 for debug)")
    p.add_argument("--background", choices=["white", "black"], default="white")
    p.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True,
                   help="Write pyramidal BigTIFF (default: True)")
    p.add_argument("--preview-max", type=int, default=1600, help="Longest edge for preview PNG")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                   help="Overwrite existing outputs (default: False)")
    p.add_argument("--px-x-um", type=float, default=None, help="Override pixel size X (µm/px)")
    p.add_argument("--px-y-um", type=float, default=None, help="Override pixel size Y (µm/px)")
    p.add_argument("--vips-workers", type=int, default=2, help="VIPS_CONCURRENCY (best effort on Windows)")

    args = p.parse_args()

    # set VIPS workers (best effort; needs to be in env before import, but we try again)
    os.environ["VIPS_CONCURRENCY"] = str(args.vips_workers)

    ndpath = Path(args.ndtiff_folder)
    if not ndpath.is_dir():
        raise SystemExit(f"Folder not found: {ndpath}")

    # input data:
    csv_path = ndpath / "summarytable.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    df = ensure_columns(df)

    # dataset & geometry
    ds = Dataset(str(ndpath))
    images = ds.as_array()
    summary = ds.summary_metadata

    px_x_um, px_y_um = parse_affine_px(summary, args.px_x_um, args.px_y_um)
    um_to_px_x, um_to_px_y = um_to_px_factory(px_x_um, px_y_um)

    # number of z-layers (prefer summary; fallback to df)
    try:
        slices = summary.get("Slices") or (images.shape[3] if hasattr(images, "shape") and images.ndim >= 4 else 1)
        S = int(slices)
    except Exception:
        S = int(df["Z_layer"].max()) + 1

    # canvas size (native), then scale
    x_max_um = float((df["X_rel_um"] + df.get("TileWidth_um", 0)).max())
    y_max_um = float((df["Y_rel_um"] + df.get("TileHeight_um", 0)).max())
    W_native = int(math.ceil(um_to_px_x(x_max_um)))
    H_native = int(math.ceil(um_to_px_y(y_max_um)))

    SCALE = float(args.scale)
    W = max(1, int(round(W_native * SCALE)))
    H = max(1, int(round(H_native * SCALE)))
    print(f"Canvas (native): {W_native}×{H_native}  -> scale={SCALE}  =>  {W}×{H}")

    # output dir
    out_dir = Path(args.out_dir) if args.out_dir else ndpath
    out_dir.mkdir(parents=True, exist_ok=True)

    # VIPS cache can spike; limit it a bit (optional)
    try:
        pyvips.cache_set_max_mem(1_000_000_000)  # ~1 GB
    except Exception:
        pass

    # background
    bgval = 255 if args.background.lower() == "white" else 0
    bg_rgb = [bgval, bgval, bgval]

    # per-Z stitching
    for z in range(S):
        dfz = df[df["Z_layer"] == z]
        if dfz.empty:
            print(f"[warn] No rows for Z={z}; skipping.")
            continue

        bigtiff_path = out_dir / f"mosaic_Z{z}.tif"
        preview_path = out_dir / f"mosaic_Z{z}_preview.png"

        if bigtiff_path.exists() and not args.overwrite:
            print(f"[skip] {bigtiff_path} exists.")
            # create preview if missing
            if not preview_path.exists():
                try:
                    img = pyvips.Image.new_from_file(str(bigtiff_path), access="sequential")
                    s_prev = min(1.0, args.preview_max / max(img.width, img.height))
                    prev = img.resize(s_prev) if s_prev < 1.0 else img
                    prev.pngsave(str(preview_path), compression=6)
                    print(f"✅ Preview (from existing): {preview_path}")
                except Exception as e:
                    print(f"[warn] Could not make preview from existing TIFF: {e}")
            continue

        # probe first tile to set bit depth
        first_row = dfz.iloc[0]
        pos0 = int(first_row["PosIndex"]) if "PosIndex" in dfz.columns else (int(first_row["Index"]) // S)
        a0 = load_tile_pos_z(images, pos0, z)
        _, fmt, _ = tile_to_vips(a0)

        # start background canvas
        mosaic = pyvips.Image.black(W, H, bands=3).cast(fmt).copy(interpretation="srgb")
        mosaic = mosaic + bg_rgb

        print(f"\n=== Z{z}: composing {len(dfz)} tiles on {W}×{H} canvas ===")
        for r in tqdm(dfz.itertuples(index=False), total=len(dfz), desc=f"Z{z} compose"):
            pos = int(getattr(r, "PosIndex", int(r.Index) // S))
            x0 = int(round(um_to_px_x(float(r.X_rel_um)) * SCALE))
            y0 = int(round(um_to_px_y(float(r.Y_rel_um)) * SCALE))

            a = load_tile_pos_z(images, pos, z)
            t, _, _ = tile_to_vips(a)
            if SCALE != 1.0:
                t = t.resize(SCALE, kernel="linear")
            mosaic = mosaic.insert(t, x0, y0)

        # write pyramidal BigTIFF (VIPS_PROGRESS=1 will print progress)
        mosaic.tiffsave(
            str(bigtiff_path),
            tile=True,
            compression="lzw",
            bigtiff=True,
            pyramid=args.pyramid,
            predictor=True,
        )
        print(f"✅ BigTIFF written: {bigtiff_path}")

        # preview
        s_prev = min(1.0, args.preview_max / max(W, H))
        prev = mosaic.resize(s_prev) if s_prev < 1.0 else mosaic
        prev.pngsave(str(preview_path), compression=6)
        print(f"✅ Preview PNG:   {preview_path}")

    print("All done.")


if __name__ == "__main__":
    main()
