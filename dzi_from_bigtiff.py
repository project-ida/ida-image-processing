#!/usr/bin/env python3

# one-time: pip install "pyvips[binary]" tqdm matplotlib
# example call:
#   python make_dzi.py "C:/path/to/folder" --tile 512 --overlap 1 --jpeg-q 90 \
#       --split-channels false --workers 2 --show-hist

import os, re, time, argparse
from pathlib import Path

# --- set libvips env BEFORE importing pyvips ---
os.environ.setdefault("VIPS_CONCURRENCY", "2")  # worker threads (best-effort on Windows)
os.environ.setdefault("VIPS_PROGRESS",   "1")   # print low-level progress, too

from tqdm.auto import tqdm as _tqdm   # noqa: E402
import pyvips                         # noqa: E402

# Optional plotting (import lazily later if not requested)
plt = None

# --- add near the top, next to other helpers ---
def open_native(tif_path: Path) -> pyvips.Image:
    """Open BigTIFF with random access, no type or colour changes."""
    return pyvips.Image.new_from_file(str(tif_path), access="random", page=0)

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
        dt = np.float32
    else:
        dt = np.dtype(_VIPS2NP[fmt])
    buf = img.write_to_memory()
    return np.ndarray(buffer=buf, dtype=dt, shape=(img.height, img.width, img.bands))

def plot_native_hist(img_native: pyvips.Image, title: str, logy: bool = True) -> bool:
    """
    Plot histogram in the image's *native* domain.
    Returns True if a native-domain plot was shown, else False (caller may fallback).
    - For integer images, uses vips.hist_find() (streaming; no full read).
    - Special-case 'ushort' (common for SEM): x-axis = 0..65535.
    """
    # single band for brightness
    band0 = img_native.extract_band(0) if img_native.bands > 1 else img_native

    # only do native hist for integer types; float histograms can be misleading without binning
    if band0.format not in ("uchar","char","ushort","short","uint","int"):
        return False

    h = band0.hist_find()  # integer-domain histogram; width == number of representable values
    counts = _vips_to_numpy(h).reshape(-1)

    import matplotlib.pyplot as plt
    xs = None
    if band0.format == "ushort":
        xs = list(range(0, counts.size))            # 0..65535
        xlab = "Value (0..65535)"
    elif band0.format == "uchar":
        xs = list(range(0, counts.size))            # 0..255
        xlab = "Value (0..255)"
    else:
        # Signed / other integer: annotate with observed min/max
        try:
            vmin, vmax = int(band0.min()), int(band0.max())
        except Exception:
            vmin, vmax = 0, counts.size - 1
        xs = list(range(0, counts.size))
        xlab = f"Value index (observed ~{vmin}..{vmax})"

    plt.figure(figsize=(8, 3.5))
    plt.bar(xs, counts, width=1.0)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.tight_layout()
    plt.show()
    return True

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


# ---------- image helpers ----------
def open_as_u8(tif_path: Path) -> pyvips.Image:
    """
    Open BigTIFF with random access and coerce to 8-bit.
    Leaves band count intact; sets sensible interpretation:
      - 1 band  -> b-w
      - 3+ bands -> srgb (if not already)
    """
    img = pyvips.Image.new_from_file(str(tif_path), access="random", page=0)

    # convert to 8-bit
    if img.format == "ushort":
        img = (img >> 8).cast("uchar")
    elif img.format != "uchar":
        img = img.cast("uchar")

    # normalize interpretation
    if img.bands == 1:
        img = img.copy(interpretation="b-w")
    elif img.bands >= 3 and img.interpretation != "srgb":
        try:
            img = img.colourspace("srgb")
        except Exception:
            img = img.copy(interpretation="srgb")

    return img.copy_memory()  # detach from on-disk file


# ---------- histogram helpers ----------
_VIPS2NP = {
    "uchar":  "uint8",
    "char":   "int8",
    "ushort": "uint16",
    "short":  "int16",
    "uint":   "uint32",
    "int":    "int32",
    "float":  "float32",
    "double": "float64",
}

def _to_numpy(img: pyvips.Image):
    """Copy a small pyvips image into a numpy array."""
    fmt = img.format
    if fmt not in _VIPS2NP:
        # fallback: cast to float
        img = img.cast("float")
        fmt = "float"
    buf = img.write_to_memory()
    np_dtype = __import__("numpy").dtype(_VIPS2NP.get(fmt, "float32"))
    import numpy as np
    arr = np.ndarray(buffer=buf, dtype=np_dtype, shape=(img.height, img.width, img.bands))
    return arr

def plot_u8_hist(img_u8: pyvips.Image, title: str, logy: bool = True):
    """
    Plot histogram for an 8-bit image (single band: take band 0).
    Uses vips.hist_find() for speed and very low memory.
    """
    global plt
    if plt is None:
        import matplotlib.pyplot as plt  # lazy import
    # pick one band (treat as brightness)
    band0 = img_u8.extract_band(0) if img_u8.bands > 1 else img_u8
    h = band0.hist_find()  # 256-bin histogram
    counts = _to_numpy(h).reshape(-1)
    xs = list(range(counts.size))
    plt.figure(figsize=(8, 3.5))
    plt.bar(xs, counts, width=1.0)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Value (0..255)")
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.tight_layout()
    plt.show()


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
    p.add_argument("--show-hist", action=argparse.BooleanOptionalAction, default=False,
                   help="Show histograms (input u8 view, output u8 to dzsave)")
    p.add_argument("--hist-logy", action=argparse.BooleanOptionalAction, default=True,
                   help="Use log scale on histogram Y axis (default true)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["VIPS_CONCURRENCY"] = str(args.workers)

    in_dir  = Path(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

        # Open + coerce to 8-bit, set interpretation
        # inside: for tif in tifs:
        img_native = open_native(tif)     # <-- NEW: native read (no coercion)
        img8       = open_as_u8(tif)      # existing: u8 view used for dzsave
        
        # quick min/max on native (helpful to see true domain)
        try:
            nmin, nmax = int(img_native.min()), int(img_native.max())
            print(f"{tif.name}: native range {nmin}..{nmax} ({img_native.format}), "
                  f"bands={img_native.bands}, {img_native.width}×{img_native.height}")
        except Exception:
            pass
        
        # --- INPUT histogram: native domain if possible; else fallback to u8 view ---
        if args.show_hist:
            shown = plot_native_hist(img_native, title=f"[INPUT/native] {tif.name}", logy=bool(args.hist_logy))
            if not shown:
                # fallback: show the u8 coerced view if native is non-integer
                plot_u8_hist(img8, title=f"[INPUT/u8 view] {tif.name}", logy=bool(args.hist_logy))

        wrote_any = False

        # --- full-color DZI (only when RGB-ish) ---
        if args.full_color and img8.bands >= 3:
            outbase_rgb = str(out_dir / f"{tif.stem}_rgb")
            if args.skip_if_exists and Path(outbase_rgb + ".dzi").exists():
                print(f"[skip] {outbase_rgb}.dzi exists")
            else:
                # histogram of what we actually send out (srgb)
                if args.show_hist:
                    plot_u8_hist(img8.copy(interpretation="srgb"),
                                 title=f"[OUTPUT] {tif.name} → RGB DZI (pre-write u8)", logy=bool(args.hist_logy))

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

                    if args.show_hist:
                        plot_u8_hist(ch, title=f"[OUTPUT] {tif.name} → {name} DZI (pre-write u8)", logy=bool(args.hist_logy))

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

        # --- guaranteed grayscale fallback (bands==1, or nothing else wrote) ---
        if not wrote_any:
            outbase_gray = str(out_dir / f"{tif.stem}_gray")
            if args.skip_if_exists and Path(outbase_gray + ".dzi").exists():
                print(f"[skip] {outbase_gray}.dzi exists")
            else:
                gray = img8.extract_band(0).copy(interpretation="b-w") if img8.bands > 1 else img8

                if args.show_hist:
                    plot_u8_hist(gray, title=f"[OUTPUT] {tif.name} → Gray DZI (pre-write u8)", logy=bool(args.hist_logy))

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
