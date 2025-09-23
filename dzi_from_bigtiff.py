#!/usr/bin/env python3
# one-time: pip install "pyvips[binary]" tqdm matplotlib

import os, re, time, argparse
from pathlib import Path

# --- set libvips env BEFORE importing pyvips ---
os.environ.setdefault("VIPS_CONCURRENCY", "2")  # worker threads (best-effort)
os.environ.setdefault("VIPS_PROGRESS",   "1")   # print low-level progress

from tqdm.auto import tqdm as _tqdm   # noqa: E402
import pyvips                         # noqa: E402
import numpy as np                    # noqa: E402

# We import matplotlib lazily when histograms are requested.
plt = None

# ---------------------------- helpers: open images -----------------------------
def open_native(tif_path: Path) -> pyvips.Image:
    """Open BigTIFF with random access, no type or colour changes."""
    return pyvips.Image.new_from_file(str(tif_path), access="random", page=0)

def open_as_u8(tif_path: Path) -> pyvips.Image:
    """
    Open BigTIFF and coerce to 8-bit (what dzsave will get).
    - ushort -> right shift 8 (0..65535 -> 0..255), then cast to uchar
    - other types -> cast to uchar
    - set sensible interpretation
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

    return img.copy_memory()  # detach from on-disk file

# ----------------------- helpers: vips<->numpy + hist plots --------------------
_VIPS2NP = {
    "uchar":  "uint8",  "char":   "int8",
    "ushort": "uint16", "short":  "int16",
    "uint":   "uint32", "int":    "int32",
    "float":  "float32","double": "float64",
}
def _vips_to_numpy(img: pyvips.Image) -> np.ndarray:
    fmt = img.format
    if fmt not in _VIPS2NP:
        img = img.cast("float")
        fmt = "float"
    buf = img.write_to_memory()
    return np.ndarray(buffer=buf,
                      dtype=np.dtype(_VIPS2NP[fmt]),
                      shape=(img.height, img.width, img.bands))

def _lazy_import_matplotlib(backend_agg: bool = True):
    global plt
    if plt is None:
        if backend_agg:
            os.environ.setdefault("MPLBACKEND", "Agg")  # safe for headless
        import matplotlib.pyplot as _plt
        plt = _plt

def plot_native_hist(img_native: pyvips.Image, title: str, logy: bool = True) -> bool:
    """
    Plot histogram in the image's native domain.
    - For integer images, use vips.hist_find() (exact integer bins).
    - For non-integer (float/double), bin between observed min..max.
    Returns True if we produced a native-domain plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    band0 = img_native.extract_band(0) if img_native.bands > 1 else img_native
    fmt = band0.format

    # Integer types -> exact integer histogram
    if fmt in ("uchar","char","ushort","short","uint","int"):
        h = band0.hist_find()
        counts = _vips_to_numpy(h).reshape(-1)
        xs = np.arange(counts.size)
        if fmt == "ushort":
            xlab = "Value (0..65535)"
        elif fmt == "uchar":
            xlab = "Value (0..255)"
        else:
            # signed/other ints: show the actual span we see
            try:
                vmin, vmax = int(band0.min()), int(band0.max())
            except Exception:
                vmin, vmax = 0, counts.size - 1
            xlab = f"Value (observed {vmin}..{vmax})"
        plt.figure(figsize=(8, 3.5))
        plt.bar(xs, counts, width=1.0)
        if logy: plt.yscale("log")
        plt.title(title)
        plt.xlabel(xlab); plt.ylabel("Count" + (" (log)" if logy else ""))
        plt.tight_layout(); plt.savefig(f"{Path(title.split()[-1]).stem}_hist_input_native.png", dpi=120)
        plt.close()
        return True

    # Float/double -> bin over observed range (no hard-coded 0..65535)
    try:
        vmin, vmax = float(band0.min()), float(band0.max())
    except Exception:
        return False
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        return False

    # Pull a downsampled view to keep it light
    # (shrink to ~4 MP if huge)
    scale = max(band0.width * band0.height / (4_000_000.0), 1.0)
    if scale > 1.0:
        f = 1.0 / math.sqrt(scale)
        band_s = band0.resize(f)  # area/nearest-ish is fine for hist
    else:
        band_s = band0

    arr = _vips_to_numpy(band_s)[:, :, 0].astype(np.float32, copy=False)
    bins = min(4096, max(512, int(math.sqrt(arr.size))))  # adaptive bins
    counts, edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(8, 3.5))
    plt.plot(centers, counts, drawstyle="steps-mid")
    if logy: plt.yscale("log")
    plt.title(f"{title}  ({fmt}, bins={bins})")
    plt.xlabel(f"Value ({vmin:.3g}..{vmax:.3g})")
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.tight_layout()
    plt.savefig(f"{Path(title.split()[-1]).stem}_hist_input_native.png", dpi=120)
    plt.close()
    return True

def plot_u8_hist(img_u8: pyvips.Image, title: str, out_png: Path, logy: bool = True) -> None:
    """Exact 8-bit histogram with fixed 0..255 axis; save to out_png."""
    _lazy_import_matplotlib()

    band0 = img_u8.extract_band(0) if img_u8.bands > 1 else img_u8
    h = band0.hist_find()               # 256 bins guaranteed
    counts = _vips_to_numpy(h).reshape(-1)[:256]
    xs = np.arange(256)

    plt.figure(figsize=(9, 3.5))
    plt.bar(xs, counts, width=1.0)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Value (0..255)")
    plt.ylabel("Count" + (" (log)" if logy else ""))
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    try:
        plt.close()
    except Exception:
        pass

# --------------------------- progress-enabled dzsave ---------------------------
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

# ----------------------------------- CLI --------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create DZI (DeepZoom) pyramids from mosaics in a folder.")
    p.add_argument("folder", help="Folder containing input .tif files")
    p.add_argument("--pattern", default="*_Z*.tif", help="Glob for input files (default '*_Z*.tif')")
    p.add_argument("--out-dir", default=None, help="Output dir for DZI (default: same as input folder)")
    p.add_argument("--tile", type=int, default=512, help="DZI tile size (default 512)")
    p.add_argument("--overlap", type=int, default=1, help="DZI tile overlap (default 1)")
    p.add_argument("--jpeg-q", type=int, default=90, help="JPEG quality if suffix=jpg (default 90)")
    p.add_argument("--png-compress", type=int, default=3, help="PNG compression level 0–9 (default 3)")
    p.add_argument("--suffix", choices=["jpg", "png"], default="jpg", help="Tile format for DZI tiles")
    p.add_argument("--full-color", action=argparse.BooleanOptionalAction, default=True,
                   help="Write RGB DZI when bands>=3 (default true)")
    p.add_argument("--split-channels", action=argparse.BooleanOptionalAction, default=False,
                   help="Also write per-channel DZIs (R/G/B or single gray)")
    p.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip if .dzi already exists (default true)")
    p.add_argument("--workers", type=int, default=2, help="libvips concurrency (best-effort)")
    # histogram options
    p.add_argument("--show-hist", action=argparse.BooleanOptionalAction, default=False,
                   help="Generate & save histogram PNGs for input(native) and output(u8)")
    p.add_argument("--hist-logy", action=argparse.BooleanOptionalAction, default=True,
                   help="Use log scale on histogram Y axis (default true)")
    p.add_argument("--hist-float-bins", type=int, default=2048,
                   help="Bins when native is float and not mappable to 0..65535 (default 2048)")
    return p.parse_args()

# ----------------------------------- main -------------------------------------
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

        # Open both: native (no coercion) and u8 (what we'll feed to dzsave)
        img_native = open_native(tif)
        img8       = open_as_u8(tif)

        # Print native quick range
        try:
            nmin, nmax = int(img_native.min()), int(img_native.max())
            print(f"{tif.name}: native range {nmin}..{nmax} ({img_native.format}), "
                  f"bands={img_native.bands}, {img_native.width}×{img_native.height}")
        except Exception:
            pass

        # --- Histograms (saved as PNG) ---
        if args.show_hist:
            _lazy_import_matplotlib(backend_agg=True)
            # INPUT native-domain hist
            input_hist_png = out_dir / f"{tif.stem}_hist_input_native.png"
            plot_native_hist(
                img_native,
                title=f"[INPUT native] {tif.name}  ({img_native.format})",
                out_png=input_hist_png,
                logy=bool(args.hist_logy),
                float_bins=int(args.hist_float_bins),
            )
            # OUTPUT u8 hist (what we actually pass to dzsave)
            output_hist_png = out_dir / f"{tif.stem}_hist_output_u8.png"
            plot_u8_hist(
                img8,
                title=f"[OUTPUT u8] {tif.stem}",
                out_png=output_hist_png,
                logy=bool(args.hist_logy),
            )

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

        # --- guaranteed grayscale fallback (bands==1, or nothing else wrote) ---
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
