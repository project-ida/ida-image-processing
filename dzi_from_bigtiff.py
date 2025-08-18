#!/usr/bin/env python3

# one-time: pip install "pyvips[binary]" tqdm
# example call: python make_dzi.py "C:/Users/MITLENR/leica-images/test-fm/test2full3z_1" --tile 512 --overlap 1 --jpeg-q 90 --split-channels false --workers 2

import os, re, time, argparse
from pathlib import Path

# --- set libvips env BEFORE importing pyvips ---
os.environ.setdefault("VIPS_CONCURRENCY", "2")  # worker threads (best-effort on Windows)
os.environ.setdefault("VIPS_PROGRESS",   "1")   # print low-level progress, too

from tqdm.auto import tqdm as _tqdm   # noqa: E402
import pyvips                         # noqa: E402


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
def open_as_u8_srgb(tif_path: Path) -> pyvips.Image:
    """Open BigTIFF and coerce to 8-bit sRGB (or b-w for single bands)."""
    # use random access + page=0 for stability with multi-page pyramids
    img = pyvips.Image.new_from_file(str(tif_path), access="random", page=0)

    # Downshift if 16-bit; otherwise ensure uchar
    if img.format == "ushort":
        img = (img >> 8).cast("uchar")
    elif img.format != "uchar":
        img = img.cast("uchar")

    # Make sure colourspace is SRGB for RGB sources
    if img.bands == 3 and img.interpretation != "srgb":
        try:
            img = img.colourspace("srgb")
        except Exception:
            img = img.copy(interpretation="srgb")

    # Detach from file so dzsave can stream independently
    return img.copy_memory()


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Create DZI (DeepZoom) pyramids from mosaics (mosaic_Z*.tif) in a folder."
    )
    p.add_argument("folder", help="Folder containing mosaic_Z*.tif")
    p.add_argument("--out-dir", default=None,
                   help="Output dir for DZI (default: same as input folder)")
    p.add_argument("--tile", type=int, default=512, help="DZI tile size (default 512)")
    p.add_argument("--overlap", type=int, default=1, help="DZI tile overlap (default 1)")
    p.add_argument("--jpeg-q", type=int, default=90,
                   help="JPEG quality if using JPEG suffix (default 90)")
    p.add_argument("--png-compress", type=int, default=3,
                   help="PNG compression level if using PNG suffix (0–9, default 3)")
    p.add_argument("--suffix", choices=["jpg", "png"], default="jpg",
                   help="Tile format for DZI tiles (jpg or png)")
    p.add_argument("--full-color", action=argparse.BooleanOptionalAction, default=True,
                   help="Write RGB DZI (default true)")
    p.add_argument("--split-channels", action=argparse.BooleanOptionalAction, default=False,
                   help="Also write per-channel DZIs (R/G/B)")
    p.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip if .dzi already exists (default true)")
    p.add_argument("--workers", type=int, default=2, help="libvips concurrency (best-effort)")
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
    tifs = sorted(in_dir.glob("mosaic_Z*.tif"))
    if not tifs:
        print(f"[warn] No BigTIFFs found in {in_dir} (expected mosaic_Z*.tif)")
        return

    for tif in tifs:
        m = re.search(r"_Z(\d+)\.tif$", tif.name, flags=re.I)
        zlabel = m.group(1) if m else tif.stem

        # Open + coerce to 8-bit SRGB
        big8 = open_as_u8_srgb(tif)

        # quick min/max to catch all-white/all-black
        try:
            mn, mx = big8.min(), big8.max()
            print(f"{tif.name}: range {mn}..{mx}, bands={big8.bands}, {big8.width}×{big8.height}")
        except Exception:
            pass

        # --- full-color DZI ---
        if args.full_color and big8.bands >= 3:
            outbase_rgb = str((out_dir / tif.stem)) + "_rgb"
            if args.skip_if_exists and Path(outbase_rgb + ".dzi").exists():
                print(f"[skip] {outbase_rgb}.dzi exists")
            else:
                dzsave_with_progress(
                    big8.copy(interpretation="srgb"),
                    outbase_rgb,
                    desc=f"DZI RGB Z{zlabel}",
                    tile_size=args.tile,
                    overlap=args.overlap,
                    suffix=suffix,
                    centre=True,
                    layout="dz",
                    strip=True,
                )

        # --- optional per-channel DZIs ---
        if args.split_channels:
            bands = min(3, big8.bands)
            names = ["R", "G", "B"][:bands]
            for b, name in enumerate(names):
                outbase_ch = str(out_dir / f"{tif.stem}_{name}")
                if args.skip_if_exists and Path(outbase_ch + ".dzi").exists():
                    print(f"[skip] {outbase_ch}.dzi exists")
                    continue
                ch = big8.extract_band(b).copy(interpretation="b-w")
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

    print("All done.")

if __name__ == "__main__":
    main()
