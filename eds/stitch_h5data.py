#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stitch SEM tiles from NPZ (h5->npz) into a single 16-bit pyramidal BigTIFF
plus an 8-bit preview PNG.

Changes vs. previous version:
- Use ONE global um_per_px for both axes (from --um-per-px or CSV).
- Pixel offsets: x_px = round(X_rel_um / um_per_px), y_px = round(Y_rel_um / um_per_px).
"""

import os, sys, csv, math, time, argparse
from pathlib import Path
import numpy as np
import pyvips
from tqdm.auto import tqdm

# ---------- helpers ----------

def sample_npz_percentiles(npz_paths, p, sample_per_tile=5000, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for pth in npz_paths:
        try:
            a = np.load(pth)["sem_data"]
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
    Optional:
      invertx,inverty,global_lo,global_hi
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
    try: return float(x)
    except Exception: return default

def parse_int(x, default=None):
    try: return int(round(float(x)))
    except Exception: return default

def np_to_vips_u16(arr: np.ndarray) -> pyvips.Image:
    a = arr
    if a.dtype == np.uint16:
        a16 = a
    elif a.dtype == np.int16:
        a16 = a.astype(np.uint16, copy=False) if a.min() >= 0 else np.clip(a, 0, 65535).astype(np.uint16, copy=False)
    elif a.dtype == np.uint8:
        a16 = (a.astype(np.uint16) * 257)
    else:
        a16 = np.clip(a.astype(np.float64), 0, 65535).astype(np.uint16)
    h, w = a16.shape
    mem = a16.tobytes()
    img = pyvips.Image.new_from_memory(mem, w, h, 1, format="ushort")
    return img.copy(interpretation="b-w")

def build_canvas_u16(width: int, height: int) -> pyvips.Image:
    return pyvips.Image.black(width, height, bands=1).cast("ushort").copy(interpretation="b-w")

def insert_tile(canvas: pyvips.Image, tile: pyvips.Image, x: int, y: int) -> pyvips.Image:
    return canvas.insert(tile, x, y, expand=True)

def save_bigtiff_u16(img: pyvips.Image, out_path: Path, tile=128, workers=2, overwrite=False):
    os.environ["VIPS_CONCURRENCY"] = str(int(workers))
    tiff_opts = dict(
        compression="deflate",
        tile=True, tile_width=tile, tile_height=tile,
        pyramid=True,
        bigtiff=True,
        strip=True
    )
    if (not overwrite) and out_path.exists():
        print(f"[skip] {out_path} exists (use --overwrite to replace)")
        return
    img.tiffsave(str(out_path), **tiff_opts)

def scale_to_u8(arr16: np.ndarray, lo: float, hi: float, gamma: float = 1.0) -> np.ndarray:
    a = arr16.astype(np.float32)
    if hi <= lo: hi = lo + 1.0
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo)
    if gamma and abs(gamma - 1.0) > 1e-6:
        a = np.power(a, float(gamma))
    return (a * 255.0 + 0.5).astype(np.uint8)

def save_preview_png(img16: pyvips.Image, out_png: Path, lo: float, hi: float, gamma: float = 1.0,
                     target_max_side: int = 1600, overwrite: bool = True):
    if (not overwrite) and out_png.exists():
        print(f"[skip] {out_png} exists")
        return
    arr = np.ndarray(
        buffer=img16.write_to_memory(),
        dtype=np.uint16,
        shape=(img16.height, img16.width, img16.bands),
    )[:, :, 0]
    arr8 = scale_to_u8(arr, lo, hi, gamma=gamma)
    h, w = arr8.shape
    if max(w, h) > target_max_side:
        scale = target_max_side / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        import PIL.Image as pil
        arr8 = np.array(pil.fromarray(arr8, mode="L").resize((new_w, new_h), pil.BILINEAR))
    out = pyvips.Image.new_from_memory(arr8.tobytes(), arr8.shape[1], arr8.shape[0], 1, "uchar")
    out = out.copy(interpretation="b-w")
    out.pngsave(str(out_png), compression=3, strip=True)

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Stitch NPZ SEM tiles into a 16-bit pyramidal BigTIFF + 8-bit preview.")
    p.add_argument("folder", help="Folder that contains summary_table.csv and NPZ files")
    p.add_argument("--um-per-px", type=float, default=None,
                   help="Global µm per pixel (if omitted, read from CSV first row and require X==Y).")
    p.add_argument("--scale", type=float, default=1.0, help="Extra scale factor (default 1.0)")
    p.add_argument("--tile", type=int, default=128, help="TIFF tile size (default 128)")
    p.add_argument("--workers", type=int, default=2, help="libvips worker threads")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                   help="Overwrite existing outputs (default true)")
    # preview normalization
    p.add_argument("--norm", choices=["none", "absolute", "global", "scan", "auto"], default="global",
                   help="Preview windowing (TIFF stays 16-bit raw)")
    p.add_argument("--lo", type=float, default=None)
    p.add_argument("--hi", type=float, default=None)
    p.add_argument("--clip-percent", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--preview-side", type=int, default=1600)
    p.add_argument("--out-dir", default=None)
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

    # Gather NPZ paths
    npz_paths = [r["npz_path"] for r in rows if r.get("npz_path") and os.path.isfile(r["npz_path"])]
    if not npz_paths:
        raise SystemExit("No NPZ files found via 'npz_path' column.")

    # ---- Global µm/px (single constant) ----
    if args.um_per_px is not None:
        um_per_px = float(args.um_per_px)
    else:
        # infer from first row; require X==Y across dataset
        x0 = parse_float(rows[0]["px_x_um"])
        y0 = parse_float(rows[0]["px_y_um"])
        if not (np.isfinite(x0) and np.isfinite(y0)):
            raise SystemExit("[error] px_x_um/px_y_um invalid in CSV first row")
        if abs(x0 - y0) > 1e-6:
            raise SystemExit(f"[error] First row suggests anisotropy (px_x_um={x0}, px_y_um={y0}). "
                             f"Provide --um-per-px explicitly if you know they should be equal.")
        # also sanity-check the rest
        for r in rows[1:]:
            xi = parse_float(r["px_x_um"])
            yi = parse_float(r["px_y_um"])
            if abs(xi - x0) > 1e-6 or abs(yi - y0) > 1e-6:
                raise SystemExit("[error] CSV indicates varying µm/px across tiles; "
                                 "use your old stitcher or pass a fixed --um-per-px if you know it’s constant.")
        um_per_px = float(x0)

    # ---- Build placement lists ----
    has_invx = "invertx" in rows[0]
    has_invy = "inverty" in rows[0]

    Xum, Yum, Wpx, Hpx, NPZ = [], [], [], [], []
    for r in rows:
        invx = parse_int(r.get("invertx", 0)) if has_invx else 0
        invy = parse_int(r.get("inverty", 0)) if has_invy else 0

        x_um = parse_float(r["X_rel_um"], 0.0)
        y_um = parse_float(r["Y_rel_um"], 0.0)
        if invx == 1: x_um = -x_um
        if invy == 1: y_um = -y_um

        Xum.append(x_um)
        Yum.append(y_um)
        Wpx.append(parse_int(r["width_px"], 0))
        Hpx.append(parse_int(r["height_px"], 0))
        NPZ.append(r["npz_path"])

    Xum = np.asarray(Xum, dtype=np.float64)
    Yum = np.asarray(Yum, dtype=np.float64)
    Wpx = np.asarray(Wpx, dtype=np.int32)
    Hpx = np.asarray(Hpx, dtype=np.int32)

    # ---- µm -> px using ONE global um_per_px (optionally scaled) ----
    s = float(args.scale)
    x_off_px = np.round((Xum / um_per_px) * s).astype(np.int64)
    y_off_px = np.round((Yum / um_per_px) * s).astype(np.int64)

    # Rebase so minimum is (0,0)
    x_off_px -= int(x_off_px.min())
    y_off_px -= int(y_off_px.min())

    # Canvas size
    canvas_w = int((x_off_px + Wpx).max())
    canvas_h = int((y_off_px + Hpx).max())
    print(f"Canvas: {canvas_w} x {canvas_h}  |  um_per_px={um_per_px}  scale={s}")

    # Compose
    canvas = build_canvas_u16(canvas_w, canvas_h)
    bar = tqdm(total=len(NPZ), unit="tile", desc="Compose")

    # For preview windowing
    g_min = None; g_max = None
    samples = []
    rng = np.random.default_rng(0)
    SAMPLE_PER_TILE = 10000

    for i, npz_path in enumerate(NPZ):
        try:
            d = np.load(npz_path)
            a = d["sem_data"] if "sem_data" in d else d[next(iter(d.files))]
        except Exception as e:
            bar.write(f"[warn] cannot read {npz_path}: {e}")
            bar.update(1); continue

        if a.ndim == 3 and a.shape[-1] == 1: a = a[..., 0]
        amin, amax = int(a.min()), int(a.max())
        g_min = amin if g_min is None else min(g_min, amin)
        g_max = amax if g_max is None else max(g_max, amax)

        if a.size > SAMPLE_PER_TILE:
            idx = rng.choice(a.size, SAMPLE_PER_TILE, replace=False)
            samples.append(a.reshape(-1)[idx])
        else:
            samples.append(a.reshape(-1))

        vx = np_to_vips_u16(a)
        canvas = insert_tile(canvas, vx, int(x_off_px[i]), int(y_off_px[i]))
        bar.update(1)

    bar.close()
    print(f"Z0 NPZ stats -> min={g_min}, max={g_max}")

    # Save BigTIFF
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{folder.name.replace(' ', '_')}_Z0.tif"
    save_bigtiff_u16(canvas, out_tif, tile=args.tile, workers=args.workers, overwrite=args.overwrite)
    print(f"✅ BigTIFF written: {out_tif}")

    # ---- Preview normalization ----
    def pick_lo_hi_from_norm(norm_mode: str, npz_paths, global_min, global_max):
        norm = (norm_mode or "none").lower()
        if norm == "absolute":
            lo = float(args.lo) if args.lo is not None else global_min
            hi = float(args.hi) if args.hi is not None else global_max
            if hi < lo: lo, hi = hi, lo
            return lo, hi
        if norm == "global":
            p = float(args.clip_percent or 1.0)
            return sample_npz_percentiles(npz_paths, p, sample_per_tile=5000, seed=0)
        if norm == "auto":
            return global_min, global_max
        return global_min, global_max

    lo_used, hi_used = pick_lo_hi_from_norm(args.norm, npz_paths, g_min, g_max)
    out_png = out_dir / f"{folder.name.replace(' ', '_')}_Z0_preview.png"
    save_preview_png(canvas, out_png, lo=lo_used, hi=hi_used,
                     gamma=args.gamma, target_max_side=args.preview_side, overwrite=True)
    print(f"🛈 Preview lo={int(round(lo_used))}, hi={int(round(hi_used))}, gamma={args.gamma}")
    print("All done.")

if __name__ == "__main__":
    main()
