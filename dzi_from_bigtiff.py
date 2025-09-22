#!/usr/bin/env python3
# one-time: pip install "pyvips[binary]" tqdm matplotlib

import os, re, time, argparse
from pathlib import Path

# --- set libvips env BEFORE importing pyvips ---
os.environ.setdefault("VIPS_CONCURRENCY", "2")  # worker threads (best-effort)
os.environ.setdefault("VIPS_PROGRESS",   "1")   # print low-level progress

from tqdm.auto import tqdm as _tqdm   # noqa: E402
import pyvips                         # noqa: E402

# We lazily import matplotlib only when plotting is requested
_MPL_AVAILABLE = True
try:
    import matplotlib
    # Use a non-interactive backend if none is set (works on servers)
    if not matplotlib.get_backend().lower().startswith(("qt", "tk", "macosx")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    _MPL_AVAILABLE = False
    plt = None


# ---------- helpers: open images ----------
def open_native(tif_path: Path) -> pyvips.Image:
    """Open BigTIFF with random access, no type/colour changes."""
    return pyvips.Image.new_from_file(str(tif_path), access="random", page=0)

def open_as_u8_legacy(tif_path: Path) -> pyvips.Image:
    """
    Open BigTIFF and coerce to 8-bit like before (legacy).
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


# ---------- numpy bridge ----------
_VIPS2NP = {
    "uchar":  "uint8",   "char":   "int8",
    "ushort": "uint16",  "short":  "int16",
    "uint":   "uint32",  "int":    "int32",
    "float":  "float32", "double": "float64",
}

def vips_to_numpy(img: pyvips.Image):
    import numpy as np
    fmt = img.format
    if fmt not in _VIPS2NP:
        img = img.cast("float")
        fmt = "float"
    buf = img.write_to_memory()
    return np.ndarray(buffer=buf,
                      dtype=np.dtype(_VIPS2NP[fmt]),
                      shape=(img.height, img.width, img.bands))


# ---------- stats / mapping ----------
def native_min_max(img: pyvips.Image) -> tuple[int, int]:
    band0 = img.extract_band(0) if img.bands > 1 else img
    return int(band0.min()), int(band0.max())

def native_percentile_lo_hi_ushort(img: pyvips.Image, clip_percent: float) -> tuple[int, int]:
    """
    Compute lo/hi from a ushort (16-bit) image by percentiles using vips.hist_find().
    clip_percent is per tail, e.g. 1.0 -> p1 / p99.
    """
    import numpy as np
    band0 = img.extract_band(0) if img.bands > 1 else img
    h = band0.hist_find()  # for ushort: 65536-bin histogram
    counts = vips_to_numpy(h).reshape(-1)
    cdf = counts.cumsum()
    tot = int(cdf[-1]) if cdf.size else 0
    if tot <= 0:
        return native_min_max(img)
    lo_idx = int((cdf >= tot * (clip_percent / 100.0)).argmax())
    hi_idx = int((cdf >= tot * (1.0 - clip_percent / 100.0)).argmax())
    return lo_idx, hi_idx

def to_u8_linear(img: pyvips.Image, lo: float, hi: float) -> pyvips.Image:
    """Map [lo,hi] -> [0,255] on all bands, clamp outside."""
    if hi <= lo:
        hi = lo + 1.0
    scale  = 255.0 / (hi - lo)
    offset = -lo * scale
    out = img.linear(scale, offset).cast("uchar")
    return out.copy(interpretation=("b-w" if out.bands == 1 else "srgb"))


# ---------- histograms ----------
def plot_native_hist(img_native: pyvips.Image, title: str, logy: bool, out_png: Path | None):
    if not _MPL_AVAILABLE:
        return
    band0 = img_native.extract_band(0) if img_native.bands > 1 else img_native
    if band0.format not in ("uchar","char","ushort","short","uint","int"):
        # For floats we’d need binning; skip to avoid misleading plots
        return
    h = band0.hist_find()
    counts = vips_to_numpy(h).reshape(-1)
    xs = list(range(counts.size))
    plt.figure(figsize=(9, 3.6))
    plt.bar(xs, counts, width=1.0)
    if logy: plt.yscale("log")
    if band0.format == "ushort":
        xlabel = "Value (0..65535)"
    elif band0.format == "uchar":
        xlabel = "Value (0..255)"
    else:
        vmin, vmax = native_min_max(band0)
        xlabel = f"Value index (~{vmin}..{vmax})"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=120)
        plt.close()
    else:
        plt.show()

def plot_u8_hist(img_u8: pyvips.Image, title: str, logy: bool, out_png: Path | None):
    if not _MPL_AVAILABLE:
        return
    band0 = img_u8.extract_band(0) if img_u8.bands > 1 else img_u8
    h = band0.hist_find()  # 256 bins
    counts = vips_to_numpy(h).reshape(-1)
    xs = list(range(counts.size))
    plt.figure(figsize=(9, 3.6))
    plt.bar(xs, counts, width=1.0)
    if logy: plt.yscale("log")
    plt.title(title)
    plt.xlabel("Value (0..255)")
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=120)
        plt.close()
    else:
        plt.show()


# ---------- progress-enabled dzsave ----------
def _safe_disconnect(img, handle):
    for name in ("signal_disconnect", "disconnect"):
        try:
            getattr(img, name)(handle)
            return
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
    p = argparse.ArgumentParser(
        description="Create DZI (DeepZoom) pyramids from mosaics in a folder."
    )
    p.add_argument("folder", help="Folder containing input .tif files")
    p.add_argument("--pattern", default="*_Z*.tif",
                   help="Glob for input files (default: '*_Z*.tif')")
    p.add_argument("--out-dir", default=None,
                   help="Output dir for DZI (default: same as input folder)")
    p.add_argument("--tile", type=int, default=512, help="DZI tile size (default 512)")
    p.add_argument("--overlap", type=int, default=1, help="DZI tile overlap (default 1)")
    p.add_argument("--jpeg-q", type=int, default=90,
                   help="JPEG quality if using JPEG suffix (default 90)")
    p.add_argument("--png-compress", type=int, default=3,
                   help="PNG compression level (0–9, default 3)")
    p.add_argument("--suffix", choices=["jpg", "png"], default="jpg",
                   help="Tile format for DZI tiles (jpg or png)")
    p.add_argument("--full-color", action=argparse.BooleanOptionalAction, default=True,
                   help="Write RGB DZI when bands>=3 (default true)")
    p.add_argument("--split-channels", action=argparse.BooleanOptionalAction, default=False,
                   help="Also write per-channel DZIs (R/G/B or single gray)")
    p.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip if .dzi already exists (default true)")
    p.add_argument("--workers", type=int, default=2, help="libvips concurrency (best-effort)")

    # Histogram display
    p.add_argument("--show-hist", action=argparse.BooleanOptionalAction, default=False,
                   help="Show/save histograms (input native + output u8)")
    p.add_argument("--hist-logy", action=argparse.BooleanOptionalAction, default=True,
                   help="Log scale on histogram Y axis (default true)")
    p.add_argument("--hist-out", default=None,
                   help="If set, save histogram PNGs into this folder instead of showing")

    # 16-bit -> 8-bit mapping
    p.add_argument("--norm", choices=["none","range","percentile","absolute"], default="percentile",
                   help="u8 mapping: none (>>8), range (min..max), percentile (clip), absolute (lo/hi)")
    p.add_argument("--clip-percent", type=float, default=1.0,
                   help="for --norm percentile: clip this percent on each tail (default 1.0)")
    p.add_argument("--lo", type=float, default=None, help="for --norm absolute: lower bound")
    p.add_argument("--hi", type=float, default=None, help="for --norm absolute: upper bound")

    return p.parse_args()


def main():
    args = parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(args.workers)

    in_dir  = Path(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_dir = Path(args.hist_out) if args.hist_out else None
    if hist_dir:
        hist_dir.mkdir(parents=True, exist_ok=True)

    # choose suffix string for dzsave
    if args.suffix == "jpg":
        suffix = f".jpg[Q={int(args.jpeg_q)},strip=true]"
    else:
        suffix = f".png[compression={int(args.png_compress)}]"

    # discover BigTIFFs
    tifs = sorted(in_dir.glob(args.pattern))
    if not tifs:
        print(f"[warn] No BigTIFFs found in {in_dir} (pattern '{args.pattern}')")
        return

    for tif in tifs:
        m = re.search(r"_Z(\d+)\.tif$", tif.name, flags=re.I)
        zlabel = m.group(1) if m else tif.stem

        img_native = open_native(tif)

        # quick min/max (native)
        try:
            nmin, nmax = native_min_max(img_native)
            print(f"{tif.name}: native range {nmin}..{nmax} ({img_native.format}), "
                  f"bands={img_native.bands}, {img_native.width}×{img_native.height}")
        except Exception:
            nmin, nmax = 0, 0

        # Input histogram (native)
        if args.show_hist:
            out_png = (hist_dir / f"{tif.stem}_input_native.png") if hist_dir else None
            plot_native_hist(img_native, title=f"[INPUT native] {tif.name}  ({img_native.format})",
                             logy=bool(args.hist_logy), out_png=out_png)

        # Decide 16-bit -> 8-bit mapping
        norm = args.norm.lower()
        if norm == "none":
            img8 = open_as_u8_legacy(tif)
            lo_used, hi_used = None, None
        elif norm == "range":
            lo_used, hi_used = nmin, nmax
            img8 = to_u8_linear(img_native, lo_used, hi_used)
        elif norm == "absolute":
            lo_used = nmin if args.lo is None else float(args.lo)
            hi_used = nmax if args.hi is None else float(args.hi)
            if hi_used < lo_used: lo_used, hi_used = hi_used, lo_used
            img8 = to_u8_linear(img_native, lo_used, hi_used)
        else:  # percentile
            if img_native.format == "ushort":
                lo_used, hi_used = native_percentile_lo_hi_ushort(img_native, float(args.clip_percent))
            else:
                lo_used, hi_used = nmin, nmax
            img8 = to_u8_linear(img_native, lo_used, hi_used)

        if lo_used is None:
            print(f"  → u8 mapping: legacy cast (norm=none)")
        else:
            print(f"  → u8 mapping: lo={int(round(lo_used))}, hi={int(round(hi_used))} "
                  f"(norm={norm}{', clip%='+str(args.clip_percent) if norm=='percentile' else ''})")

        # Output histogram (u8 we actually write)
        if args.show_hist:
            out_png = (hist_dir / f"{tif.stem}_output_u8.png") if hist_dir else None
            plot_u8_hist(img8, title=f"[OUTPUT u8] {tif.stem}", logy=bool(args.hist_logy), out_png=out_png)

        wrote_any = False

        # --- full-color DZI (only when RGB-ish) ---
        if args.full_color and img8.bands >= 3:
            outbase_rgb = str(out_dir / f"{tif.stem}_rgb")
            if args.skip_if_exists and Path(outbase_rgb + ".dzi").exists():
                print(f"[skip] {outbase_rgb}.dzi exists")
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

        # --- per-channel DZIs ---
        if args.split_channels:
            bands = min(3, img8.bands)
            if bands == 0:
                print(f"[warn] {tif.name}: zero bands? skipping.")
            else:
                names = ["R", "G", "B"][:bands] if bands >= 3 else ["Gray"]
                for b, name in enumerate(names):
                    outbase_ch = str(out_dir / f"{tif.stem}_{name}")
                    if args.skip_if_exists and Path(outbase_ch + ".dzi").exists():
                        print(f"[skip] {outbase_ch}.dzi exists")
                        continue
                    ch = img8.extract_band(b if bands >= 3 else 0).copy(interpretation="b-w")
                    dzsave_with_progress(
                        ch,
                        outbase_ch,
                        desc=f"DZI {name} Z{zlabel}",
                        tile_size=args.tile,
                        overlap=args.overlap,
                        suffix=suffix,
                        centre=True,
                        layout="dz",
                        strip=True,
                    )
                    wrote_any = True

        # --- guaranteed grayscale fallback ---
        if not wrote_any:
            outbase_gray = str(out_dir / f"{tif.stem}_gray")
            if args.skip_if_exists and Path(outbase_gray + ".dzi").exists():
                print(f"[skip] {outbase_gray}.dzi exists")
            else:
                gray = img8.extract_band(0).copy(interpretation="b-w") if img8.bands > 1 else img8
                dzsave_with_progress(
                    gray,
                    outbase_gray,
                    desc=f"DZI Gray Z{zlabel}",
                    tile_size=args.tile,
                    overlap=args.overlap,
                    suffix=suffix,
                    centre=True,
                    layout="dz",
                    strip=True,
                )

    print("All done.")

if __name__ == "__main__":
    main()
