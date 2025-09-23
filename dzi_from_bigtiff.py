#!/usr/bin/env python3
# one-time: pip install "pyvips[binary]" tqdm matplotlib
# example call:
#   python dzi_from_bigtiff.py "C:/path/to/folder" --tile 512 --overlap 1 --jpeg-q 90 \
#       --split-channels false --workers 2 --show-hist --clip-percent 1.0

import os, re, time, argparse
from pathlib import Path

# --- set libvips env BEFORE importing pyvips ---
os.environ.setdefault("VIPS_CONCURRENCY", "2")  # worker threads (best-effort)
os.environ.setdefault("VIPS_PROGRESS",   "1")   # print low-level progress, too

from tqdm.auto import tqdm as _tqdm   # noqa: E402
import pyvips                         # noqa: E402

# We import matplotlib lazily when/if needed.
_MPL = None

# ---------- image helpers ----------
def open_native(tif_path: Path) -> pyvips.Image:
    """Open BigTIFF with random access, no type/colour changes."""
    return pyvips.Image.new_from_file(str(tif_path), access="random", page=0)

def open_as_u8(tif_path: Path) -> pyvips.Image:
    """
    Open BigTIFF and coerce to 8-bit (no stretching).
    16-bit ushorts are shifted >> 8. Interpretation is normalized.
    """
    img = pyvips.Image.new_from_file(str(tif_path), access="random", page=0)
    if img.format == "ushort":
        img = (img >> 8).cast("uchar")
    elif img.format != "uchar":
        img = img.cast("uchar")

    if img.bands == 1:
        img = img.copy(interpretation="b-w")
    elif img.bands >= 3 and img.interpretation != "srgb":
        try:
            img = img.colourspace("srgb")
        except Exception:
            img = img.copy(interpretation="srgb")
    return img.copy_memory()

# ---------- histogram + plotting ----------
def _ensure_mpl():
    global _MPL
    if _MPL is None:
        import matplotlib
        # Always use a file-capable backend
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        _MPL = plt
    return _MPL

def _save_or_show(fig, out_png: Path, show: bool):
    plt = _ensure_mpl()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"[hist] wrote {out_png}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

def plot_native_histogram(img_native: pyvips.Image, title: str, logy: bool, out_png: Path, show: bool):
    """
    Plot histogram in the *native* domain using vips.hist_find for integer formats.
    For float, we downsample to a 2048-bin histogram via vips (hist_find not applicable).
    """
    plt = _ensure_mpl()

    # brightness band
    band = img_native.extract_band(0) if img_native.bands > 1 else img_native

    # integer?
    int_formats = {"uchar","char","ushort","short","uint","int"}
    if band.format in int_formats:
        h = band.hist_find()  # width = # of representable values for that format
        counts = h.write_to_memory()
        import numpy as np
        arr = np.frombuffer(counts, dtype="uint32").reshape(h.height, h.width, h.bands)
        y = arr.reshape(-1)
        x = list(range(y.size))
        xlabel = {
            "uchar":"Value (0..255)",
            "char":"Value (-128..127)",
            "ushort":"Value (0..65535)",
            "short":"Value (-32768..32767)",
            "uint":"Value (0..2^32-1)",
            "int":"Value (~-2^31..2^31-1)",
        }.get(band.format, "Value")
        fig = plt.figure(figsize=(9,3.2))
        ax = fig.add_subplot(111)
        ax.bar(x, y, width=1.0)
        if logy: ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count" + (" (log)" if logy else ""))
        _save_or_show(fig, out_png, show)
        return

    # float: make a 2048-bin histogram via vips (fast streaming) using scale/offset
    try:
        # vips.hist_find requires integer; map floats into 0..2047 then hist
        lo, hi = float(band.min()), float(band.max())
        if hi <= lo: hi = lo + 1.0
        scale = 2047.0 / (hi - lo)
        offset = -lo * scale
        hist = band.linear(scale, offset).cast("ushort").hist_find()
        counts = hist.write_to_memory()
        import numpy as np
        y = np.frombuffer(counts, dtype="uint32").reshape(-1)
        x = list(range(y.size))
        fig = _ensure_mpl().figure(figsize=(9,3.2))
        ax = fig.add_subplot(111)
        ax.bar(x, y, width=1.0)
        if logy: ax.set_yscale("log")
        ax.set_title(f"{title} (float → 2048 bins)")
        ax.set_xlabel("Scaled bin (0..2047)")
        ax.set_ylabel("Count" + (" (log)" if logy else ""))
        _save_or_show(fig, out_png, show)
    except Exception as e:
        print(f"[hist] float histogram failed: {e}")

def plot_u8_histogram(img_u8: pyvips.Image, title: str, logy: bool, out_png: Path, show: bool):
    plt = _ensure_mpl()
    band = img_u8.extract_band(0) if img_u8.bands > 1 else img_u8
    h = band.hist_find()  # 256 bins
    counts = h.write_to_memory()
    import numpy as np
    y = np.frombuffer(counts, dtype="uint32").reshape(-1)
    x = list(range(y.size))
    fig = plt.figure(figsize=(9,3.2))
    ax = fig.add_subplot(111)
    ax.bar(x, y, width=1.0)
    if logy: ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Value (0..255)")
    ax.set_ylabel("Count" + (" (log)" if logy else ""))
    _save_or_show(fig, out_png, show)

# ---------- percentile mapping ----------
def compute_percentile_window(img_native: pyvips.Image, clip_percent: float = 1.0) -> tuple[float,float]:
    """
    Compute (lo, hi) percentiles from the native image using streaming stats.
    We sample down using vips.avg to keep it fast and memory-light.
    """
    # get min/max quick (may be needed for clamping)
    lo = float(img_native.min())
    hi = float(img_native.max())
    if hi <= lo: return lo, hi

    # Build a 1-band float thumbnail and compute percentiles on that raster
    # (fast enough and robust for preview/normalization decisions).
    # Shrink to max ~2048 on long edge.
    scale = max(img_native.width, img_native.height) / 2048.0
    scale = max(1.0, scale)
    thumb = img_native.extract_band(0).shrink(scale, scale).cast("float")

    # Pull small thumb to numpy and compute percentiles
    buf = thumb.write_to_memory()
    import numpy as np
    arr = np.frombuffer(buf, dtype="float32").reshape(thumb.height, thumb.width, thumb.bands)
    a = arr.reshape(-1)
    p = float(clip_percent)
    plo, phi = np.percentile(a, [p, 100.0 - p])
    # Clamp to observed native min/max
    plo = max(plo, lo); phi = min(phi, hi)
    if phi <= plo: phi = plo + 1.0
    return float(plo), float(phi)

def apply_linear_u8(img_native: pyvips.Image, lo: float, hi: float) -> pyvips.Image:
    """Map native range [lo,hi] → u8 0..255 with clipping."""
    if hi <= lo:
        hi = lo + 1.0
    scale  = 255.0 / (hi - lo)
    offset = -lo * scale
    u8 = img_native.extract_band(0).linear(scale, offset).cast("uchar")
    return u8.copy(interpretation="b-w")

# ---------- progress-enabled dzsave ----------
def _safe_disconnect(img, handle):
    for name in ("signal_disconnect", "disconnect"):
        try:
            getattr(img, name)(handle); return
        except Exception:
            pass

def dzsave_with_progress(img: pyvips.Image, outbase: str, desc: str, **opts):
    bar = {"obj": None}
    def _pre(_img, prog):
        total = getattr(prog, "tpels", None)
        bar["obj"] = _tqdm(total=total or 0, unit="px", desc=desc, leave=True)
    def _eval(_img, prog):
        if bar["obj"]:
            done = getattr(prog, "npels", None)
            if done is not None and bar["obj"].total:
                bar["obj"].n = done
            bar["obj"].refresh()
    def _post(_img, prog):
        if bar["obj"]:
            if bar["obj"].total:
                bar["obj"].n = bar["obj"].total
            bar["obj"].close()
    img.set_progress(True)
    h1 = img.signal_connect("preeval", _pre)
    h2 = img.signal_connect("eval", _eval)
    h3 = img.signal_connect("posteval", _post)
    try:
        t0 = time.time()
        img.dzsave(outbase, **opts)
    finally:
        _safe_disconnect(img, h1); _safe_disconnect(img, h2); _safe_disconnect(img, h3)
        try: img.set_progress(False)
        except Exception: pass
    print(f"✅ {desc} -> {outbase}.dzi  ({time.time()-t0:.1f}s)")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Create DZI (DeepZoom) pyramids from mosaics in a folder.")
    p.add_argument("folder", help="Folder containing input .tif files")
    p.add_argument("--pattern", default="*_Z*.tif", help="Glob for input files (default: '*_Z*.tif')")
    p.add_argument("--out-dir", default=None, help="Output dir for DZI (default: same as input folder)")
    p.add_argument("--tile", type=int, default=512, help="DZI tile size (default 512)")
    p.add_argument("--overlap", type=int, default=1, help="DZI tile overlap (default 1)")
    p.add_argument("--jpeg-q", type=int, default=90, help="JPEG quality if using JPEG suffix (default 90)")
    p.add_argument("--png-compress", type=int, default=3, help="PNG compression level (0–9, default 3)")
    p.add_argument("--suffix", choices=["jpg", "png"], default="jpg", help="Tile format for DZI tiles (jpg or png)")
    p.add_argument("--full-color", action=argparse.BooleanOptionalAction, default=True,
                   help="Write RGB DZI when bands>=3 (default true)")
    p.add_argument("--split-channels", action=argparse.BooleanOptionalAction, default=False,
                   help="Also write per-channel DZIs (R/G/B or single gray)")
    p.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip if .dzi already exists (default true)")
    p.add_argument("--workers", type=int, default=2, help="libvips concurrency (best-effort)")
    # histogram + mapping options
    p.add_argument("--show-hist", action=argparse.BooleanOptionalAction, default=False,
                   help="Save histogram PNGs (and show if a GUI backend is available)")
    p.add_argument("--hist-logy", action=argparse.BooleanOptionalAction, default=True,
                   help="Use log scale on histogram Y axis (default true)")
    p.add_argument("--hist-dir", default=None,
                   help="Folder to write histogram PNGs (default: alongside output)")
    p.add_argument("--clip-percent", type=float, default=1.0,
                   help="Percentile clip for u8 mapping (default 1.0). Set 0 to use full min..max.")
    return p.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(args.workers)

    in_dir  = Path(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # suffix string for dzsave
    if args.suffix == "jpg":
        suffix = f".jpg[Q={int(args.jpeg_q)},strip=true]"
    else:
        suffix = f".png[compression={int(args.png_compress)}]"

    # discover BigTIFFs
    tifs = sorted(in_dir.glob(args.pattern))
    if not tifs:
        print(f"[warn] No BigTIFFs found in {in_dir} (pattern '{args.pattern}')")
        return

    # decide hist output dir
    hist_dir = Path(args.hist_dir) if args.hist_dir else (out_dir / "histograms")
    show_plots = False  # we always save; showing is best-effort if user has GUI

    for tif in tifs:
        m = re.search(r"_Z(\d+)\.tif$", tif.name, flags=re.I)
        zlabel = m.group(1) if m else tif.stem

        img_native = open_native(tif)   # 16-bit (or whatever native)
        img8_no_stretch = open_as_u8(tif)  # 8-bit view by shift/cast

        # quick native min/max
        try:
            nmin, nmax = int(img_native.min()), int(img_native.max())
            print(f"{tif.name}: native range {nmin}..{nmax} ({img_native.format}), "
                  f"bands={img_native.bands}, {img_native.width}×{img_native.height}")
        except Exception:
            pass

        # input histogram(s)
        if args.show_hist:
            plot_native_histogram(
                img_native,
                title=f"[INPUT native] {tif.name}  ({img_native.format})",
                logy=bool(args.hist_logy),
                out_png=hist_dir / f"{tif.stem}_hist_input_native.png",
                show=show_plots,
            )

        # --- u8 mapping for DZI (linear stretch with optional percentile clip) ---
        if args.clip_percent and args.clip_percent > 0:
            lo, hi = compute_percentile_window(img_native, clip_percent=float(args.clip_percent))
            print(f"→ u8 mapping: lo={int(round(lo))}, hi={int(round(hi))} (norm=percentile, clip%={args.clip_percent})")
        else:
            lo, hi = float(img_native.min()), float(img_native.max())
            print(f"→ u8 mapping: lo={int(round(lo))}, hi={int(round(hi))} (norm=full range)")

        mapped_u8 = apply_linear_u8(img_native, lo, hi)

        # output histogram(s) for what we actually feed to dzsave
        if args.show_hist:
            plot_u8_histogram(
                mapped_u8,
                title=f"[OUTPUT u8] {tif.stem}",
                logy=bool(args.hist_logy),
                out_png=hist_dir / f"{tif.stem}_hist_output_u8.png",
                show=show_plots,
            )

        wrote_any = False

        # --- full-color DZI (only when RGB-ish) ---
        if args.full_color and mapped_u8.bands >= 3:
            outbase_rgb = str(out_dir / f"{tif.stem}_rgb")
            if args.skip_if_exists and Path(outbase_rgb + ".dzi").exists():
                print(f"[skip] {outbase_rgb}.dzi exists")
            else:
                dzsave_with_progress(
                    mapped_u8.copy(interpretation="srgb"),
                    outbase_rgb,
                    desc=f"DZI RGB Z{zlabel}",
                    tile_size=args.tile, overlap=args.overlap,
                    suffix=suffix, centre=True, layout="dz", strip=True,
                )
                wrote_any = True

        # --- per-channel DZIs ---
        # (Most SEM mosaics are 1-band; keep feature for completeness.)
        if args.split_channels and mapped_u8.bands >= 3:
            names = ["R","G","B"]
            for b, name in enumerate(names):
                outbase_ch = str(out_dir / f"{tif.stem}_{name}")
                if args.skip_if_exists and Path(outbase_ch + ".dzi").exists():
                    print(f"[skip] {outbase_ch}.dzi exists"); continue
                ch = mapped_u8.extract_band(b).copy(interpretation="b-w")
                dzsave_with_progress(
                    ch,
                    outbase_ch,
                    desc=f"DZI {name} Z{zlabel}",
                    tile_size=args.tile, overlap=args.overlap,
                    suffix=suffix, centre=True, layout="dz", strip=True,
                )
                wrote_any = True

        # --- guaranteed grayscale fallback ---
        if not wrote_any:
            outbase_gray = str(out_dir / f"{tif.stem}_gray")
            if args.skip_if_exists and Path(outbase_gray + ".dzi").exists():
                print(f"[skip] {outbase_gray}.dzi exists")
            else:
                gray = mapped_u8.extract_band(0).copy(interpretation="b-w") \
                       if mapped_u8.bands > 1 else mapped_u8
                dzsave_with_progress(
                    gray,
                    outbase_gray,
                    desc=f"DZI Gray Z{zlabel}",
                    tile_size=args.tile, overlap=args.overlap,
                    suffix=suffix, centre=True, layout="dz", strip=True,
                )

    print("All done.")

if __name__ == "__main__":
    main()
