#!/usr/bin/env python3
# one-time: pip install "pyvips[binary]" tqdm matplotlib
# example:
#   python dzi_from_bigtiff.py "/path/to/folder" --tile 512 --overlap 1 \
#       --suffix jpg --jpeg-q 90 --split-channels false --workers 2 \
#       --show-hist --norm percentile --clip-percent 1.0

import os, re, time, argparse, math
from pathlib import Path

# ---- libvips env (must be set before importing pyvips) -----------------------
os.environ.setdefault("VIPS_CONCURRENCY", "2")  # worker threads (best-effort)
os.environ.setdefault("VIPS_PROGRESS",   "1")   # also print low-level progress

from tqdm.auto import tqdm as _tqdm     # noqa: E402
import pyvips                           # noqa: E402

# lazy import for plotting (only when --show-hist)
plt = None

# ---- helpers: open images ----------------------------------------------------
def open_native(tif_path: Path) -> pyvips.Image:
    """Open BigTIFF with random access without type/colour changes."""
    return pyvips.Image.new_from_file(str(tif_path), access="random", page=0)

# ---- numpy bridge for tiny vips images (histograms etc.) ---------------------
_VIPS2NP = {
    "uchar":  "uint8",  "char":   "int8",
    "ushort": "uint16", "short":  "int16",
    "uint":   "uint32", "int":    "int32",
    "float":  "float32","double": "float64",
}
def _vips_to_numpy(img: pyvips.Image):
    import numpy as np
    fmt = img.format
    if fmt not in _VIPS2NP:
        img = img.cast("float")
        fmt = "float"
    buf = img.write_to_memory()
    return __import__("numpy").ndarray(
        buffer=buf,
        dtype=__import__("numpy").dtype(_VIPS2NP[fmt]),
        shape=(img.height, img.width, img.bands),
    )

# ---- histogram plotting -------------------------------------------------------
def plot_native_hist(img_native: pyvips.Image, title: str, out_png: Path, logy: bool = True) -> bool:
    """
    Plot histogram in the image's *native* domain.
    - Integer images: exact histogram via vips.hist_find()
    - Float/double:   sampled histogram between observed min..max
    Returns True if native-domain plot was produced.
    """
    global plt
    import numpy as np
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt  # noqa: F401

    band0 = img_native.extract_band(0) if img_native.bands > 1 else img_native
    fmt = band0.format

    # Integer types -> exact integer histogram
    if fmt in ("uchar", "char", "ushort", "short", "uint", "int"):
        h = band0.hist_find()
        counts = _vips_to_numpy(h).reshape(-1)
        xs = range(counts.size)
        if fmt == "ushort":
            xlab = "Value (0..65535)"
        elif fmt == "uchar":
            xlab = "Value (0..255)"
        else:
            try:
                vmin, vmax = int(band0.min()), int(band0.max())
            except Exception:
                vmin, vmax = 0, counts.size - 1
            xlab = f"Value (observed {vmin}..{vmax})"
        plt.figure(figsize=(9, 3.6))
        plt.bar(list(xs), counts, width=1.0)
        if logy: plt.yscale("log")
        plt.title(title)
        plt.xlabel(xlab); plt.ylabel("Count" + (" (log)" if logy else ""))
        plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
        return True

    # Float/double -> bin over observed range
    try:
        vmin, vmax = float(band0.min()), float(band0.max())
    except Exception:
        return False
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return False

    # downsample a bit (to ~4MP) to keep it light
    target = 4_000_000.0
    scale = max((band0.width * band0.height) / target, 1.0)
    band_s = band0.resize(1.0 / math.sqrt(scale)) if scale > 1.0 else band0

    arr = _vips_to_numpy(band_s)[:, :, 0].astype("float32", copy=False)
    bins = min(4096, max(512, int(math.sqrt(arr.size))))
    import numpy as np
    counts, edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(9, 3.6))
    plt.plot(centers, counts, drawstyle="steps-mid")
    if logy: plt.yscale("log")
    plt.title(f"{title}  ({fmt}, bins={bins})")
    plt.xlabel(f"Value ({vmin:.3g}..{vmax:.3g})")
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
    return True

def plot_u8_hist(img_u8: pyvips.Image, title: str, out_png: Path, logy: bool = True):
    """Histogram for an 8-bit image (single band: band 0)."""
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    band0 = img_u8.extract_band(0) if img_u8.bands > 1 else img_u8
    h = band0.hist_find()
    counts = _vips_to_numpy(h).reshape(-1)
    xs = list(range(counts.size))
    plt.figure(figsize=(9, 3.6))
    plt.bar(xs, counts, width=1.0)
    if logy: plt.yscale("log")
    plt.title(title)
    plt.xlabel("Value (0..255)")
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()

# ---- percentile helpers (from histogram) -------------------------------------
def percentile_from_hist(counts, lo_p, hi_p):
    """Return (lo, hi) values (bin indices) for lo/hi percent from integer histogram."""
    import numpy as np
    c = counts.astype("float64")
    s = c.sum()
    if s <= 0:  # degenerate
        return 0.0, float(len(counts) - 1)
    cdf = np.cumsum(c) / s
    lo = float(np.searchsorted(cdf, lo_p / 100.0, side="left"))
    hi = float(np.searchsorted(cdf, 1.0 - hi_p / 100.0, side="left"))
    hi = max(lo + 1.0, hi)
    return lo, hi

def choose_window(img: pyvips.Image, norm: str, clip_percent: float, lo_abs, hi_abs):
    """
    Decide lo/hi in the *native* domain.
    - integer types: use exact hist_find() percentiles
    - float/double: sample into numpy and use np.percentile
    """
    band0 = img.extract_band(0) if img.bands > 1 else img
    fmt = band0.format

    # fixed
    if norm == "fixed":
        if lo_abs is None or hi_abs is None:
            raise SystemExit("[error] --norm fixed requires both --lo and --hi")
        lo, hi = float(lo_abs), float(hi_abs)
        if hi <= lo: lo, hi = hi, lo
        return lo, hi, "fixed"

    # minmax
    if norm == "minmax":
        lo, hi = float(band0.min()), float(band0.max())
        if hi <= lo: hi = lo + 1.0
        return lo, hi, "minmax"

    # percentile (default)
    if norm == "percentile":
        p = float(clip_percent or 0.0)
        # integers -> exact histogram
        if fmt in ("uchar", "char", "ushort", "short", "uint", "int"):
            h = band0.hist_find()
            counts = _vips_to_numpy(h).reshape(-1)
            lo, hi = percentile_from_hist(counts, p, p)
            return lo, hi, f"percentile (±{p}%)"
        # floats -> sample
        try:
            vmin, vmax = float(band0.min()), float(band0.max())
        except Exception:
            vmin, vmax = 0.0, 1.0
        target = 4_000_000.0
        scale = max((band0.width * band0.height) / target, 1.0)
        band_s = band0.resize(1.0 / math.sqrt(scale)) if scale > 1.0 else band0
        arr = _vips_to_numpy(band_s)[:, :, 0].astype("float32", copy=False)
        import numpy as np
        lo, hi = np.percentile(arr, [p, 100.0 - p])
        if hi <= lo: hi = lo + 1.0
        return float(lo), float(hi), f"percentile (±{p}%)"

    # none: pass-through domain (will later cast/shift if ushort)
    lo, hi = float(band0.min()), float(band0.max())
    if hi <= lo: hi = lo + 1.0
    return lo, hi, "none"

# ---- map native -> u8 --------------------------------------------------------
def map_to_u8(img_native: pyvips.Image, lo: float, hi: float) -> pyvips.Image:
    """
    Clip to [lo, hi] in native domain and scale to 0..255 (u8).
    Works for 1-band or multi-band (applies same lo/hi to all).
    """
    if hi <= lo:
        hi = lo + 1.0
    # vectorize to all bands
    band = img_native
    # (x - lo)
    band = band - lo
    # clip to [0, hi-lo]
    rng = hi - lo
    band = band.clip(0, rng)
    # scale to 0..255
    band = band * (255.0 / rng)
    band = band.cast("uchar")

    # set interpretation
    if band.bands == 1:
        band = band.copy(interpretation="b-w")
    elif band.bands >= 3 and band.interpretation != "srgb":
        try:
            band = band.colourspace("srgb")
        except Exception:
            band = band.copy(interpretation="srgb")
    return band

# ---- progress-enabled dzsave --------------------------------------------------
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
            if bar["obj"].total: bar["obj"].n = bar["obj"].total
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
    print(f"✅  {desc} -> {outbase}.dzi  ({time.time()-t0:.1f}s)")

# ---- CLI ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create DZI (DeepZoom) pyramids from mosaics in a folder.")
    p.add_argument("folder", help="Folder containing input .tif files")
    p.add_argument("--pattern", default="*_Z*.tif", help="Glob for input files (default: '*_Z*.tif')")
    p.add_argument("--out-dir", default=None, help="Output dir for DZI + hist PNGs (default: same as input)")
    p.add_argument("--tile", type=int, default=512, help="DZI tile size (default 512)")
    p.add_argument("--overlap", type=int, default=1, help="DZI tile overlap (default 1)")
    p.add_argument("--jpeg-q", type=int, default=90, help="JPEG quality if using JPEG suffix (default 90)")
    p.add_argument("--png-compress", type=int, default=3, help="PNG compression (0–9, default 3)")
    p.add_argument("--suffix", choices=["jpg", "png"], default="jpg", help="DZI tile format")
    p.add_argument("--full-color", action=argparse.BooleanOptionalAction, default=True,
                   help="Write RGB DZI when bands>=3 (default true)")
    p.add_argument("--split-channels", action=argparse.BooleanOptionalAction, default=False,
                   help="Also write per-channel DZIs (R/G/B or single gray)")
    p.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip if .dzi already exists (default true)")
    p.add_argument("--workers", type=int, default=2, help="libvips concurrency (best-effort)")
    p.add_argument("--show-hist", action=argparse.BooleanOptionalAction, default=False,
                   help="Export histograms: native input + u8 output")
    p.add_argument("--hist-logy", action=argparse.BooleanOptionalAction, default=True,
                   help="Use log scale on histogram Y axis (default true)")

    # NEW: normalization to map native -> u8
    p.add_argument("--norm",
                   choices=["percentile", "minmax", "fixed", "none"],
                   default="percentile",
                   help="How to window native values before 8-bit mapping (default: percentile)")
    p.add_argument("--clip-percent", type=float, default=1.0,
                   help="Clip percent for --norm percentile (default 1.0)")
    p.add_argument("--lo", type=float, default=None,
                   help="Absolute lo for --norm fixed")
    p.add_argument("--hi", type=float, default=None,
                   help="Absolute hi for --norm fixed")
    return p.parse_args()

# ---- main --------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(args.workers)

    in_dir  = Path(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # choose suffix string for dzsave
    suffix = f".jpg[Q={int(args.jpeg_q)},strip=true]" if args.suffix == "jpg" \
             else f".png[compression={int(args.png_compress)}]"

    # discover BigTIFFs
    tifs = sorted(in_dir.glob(args.pattern))
    if not tifs:
        print(f"[warn] No BigTIFFs found in {in_dir} (pattern '{args.pattern}')")
        return

    for tif in tifs:
        stem = tif.stem
        m = re.search(r"_Z(\d+)\.tif$", tif.name, flags=re.I)
        zlabel = m.group(1) if m else stem

        print(f"\n=== Processing {tif.name} ===")

        # PASS 1: open + quick scan
        print("  [PASS 1/3] Scanning input (native read, no coercion)...")
        img_native = open_native(tif)
        try:
            nmin, nmax = img_native.min(), img_native.max()
            print(f"    native range {nmin}..{nmax} ({img_native.format}), "
                  f"bands={img_native.bands}, {img_native.width}×{img_native.height}")
        except Exception:
            pass

        # Optional histogram of the *native* domain
        if args.show_hist:
            out_png = out_dir / f"{stem}_hist_input_native.png"
            shown = plot_native_hist(img_native,
                                     title=f"[INPUT native] {tif.name}",
                                     out_png=out_png,
                                     logy=bool(args.hist_logy))
            if shown:
                print(f"    wrote {out_png.name}")
            else:
                print("    [note] native histogram not available; non-finite or unsupported type")

        # Decide window and map to 8-bit
        print("  [PASS 2/3] Computing window + mapping to 8-bit for DZI...")
        lo, hi, why = choose_window(img_native, args.norm, args.clip_percent, args.lo, args.hi)
        print(f"    → u8 mapping: lo={int(round(lo))}, hi={int(round(hi))} (norm={why})")
        img8 = map_to_u8(img_native, lo, hi)

        if args.show_hist:
            out_png = out_dir / f"{stem}_hist_output_u8.png"
            # choose which image we'll histogram (gray or srgb)
            if args.full_color and img8.bands >= 3:
                _img = img8.copy(interpretation="srgb")
            else:
                _img = img8.extract_band(0).copy(interpretation="b-w") if img8.bands > 1 else img8
            plot_u8_hist(_img, title=f"[OUTPUT u8] {stem}", out_png=out_png, logy=bool(args.hist_logy))
            print(f"    wrote {out_png.name}")

        # PASS 3: write DZI(s)
        print("  [PASS 3/3] Writing DZI pyramids...")
        wrote_any = False

        # full-colour path
        if args.full_color and img8.bands >= 3:
            outbase_rgb = str(out_dir / f"{stem}_rgb")
            if args.skip_if_exists and Path(outbase_rgb + ".dzi").exists():
                print(f"    [skip] {Path(outbase_rgb + '.dzi').name} exists")
            else:
                dzsave_with_progress(
                    img8.copy(interpretation="srgb"),
                    outbase_rgb,
                    desc=f"DZI RGB Z{zlabel}",
                    tile_size=args.tile,
                    overlap=args.overlap,
                    suffix=suffix,
                    centre=True,
                    layout="dz",
                    strip=True,
                )
                wrote_any = True

        # split channels if requested
        if args.split_channels:
            bands = min(3, img8.bands)
            names = ["R", "G", "B"][:bands] if bands >= 3 else (["Gray"] if bands == 1 else [])
            for b, name in enumerate(names):
                outbase_ch = str(out_dir / f"{stem}_{name}")
                if args.skip_if_exists and Path(outbase_ch + ".dzi").exists():
                    print(f"    [skip] {Path(outbase_ch + '.dzi').name} exists")
                    continue
                ch = img8.extract_band(b if bands >= 3 else 0).copy(interpretation="b-w")
                dzsave_with_progress(
                    ch, outbase_ch, desc=f"DZI {name} Z{zlabel}",
                    tile_size=args.tile, overlap=args.overlap,
                    suffix=suffix, centre=True, layout="dz", strip=True,
                )
                wrote_any = True

        # guaranteed grayscale fallback (or if nothing else wrote)
        if not wrote_any:
            outbase_gray = str(out_dir / f"{stem}_Gray")
            if args.skip_if_exists and Path(outbase_gray + ".dzi").exists():
                print(f"    [skip] {Path(outbase_gray + '.dzi').name} exists")
            else:
                gray = img8.extract_band(0).copy(interpretation="b-w") if img8.bands > 1 else img8
                dzsave_with_progress(
                    gray, outbase_gray, desc=f"DZI Gray Z{zlabel}",
                    tile_size=args.tile, overlap=args.overlap,
                    suffix=suffix, centre=True, layout="dz", strip=True,
                )

    print("\nAll done.")

if __name__ == "__main__":
    main()
