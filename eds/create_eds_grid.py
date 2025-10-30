#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create stitching rectangles and intensity overlay JSON from summary_table.csv,
and (optionally) per-tile spectrum PNGs for tooltips.

Outputs:
  - rectangles.json
      [{id,name,x,y,width,height}]  (units in MICRONS, x/y relative; matches your old file)
  - adjusted_intensity_rectangles2.json
      [{x,y,width,height,filename,intensity}] where:
        * x,y are in PIXELS (image space, top-left origin),
        * width,height are in MICRONS,
        * filename is a PNG placed in spectraplots/ (tooltip),
        * intensity is normalized 0..1 (with optional percentile clip)
  - spectraplots/<name>.png  (one per tile; optional, controlled by --make-plots)

Assumptions:
  - summary_table.csv columns (produced by your notebook pipeline):
      basename,png_path,npz_path,width_px,height_px,px_x_um,px_y_um,
      X_rel_um,Y_rel_um[,invertx,inverty,...]
  - EDS NPZ files live alongside the SEM/NPZ or in --eds-dir and are named like:
      "<basename>_eds.npz", containing array key "eds_data" (H, W, C).
    If you already store an explicit "eds_npz_path" column, we use it.

Viewer expectations for adjusted_intensity_rectangles2.json:
  - The HTML turns (x,y) into normalized coords assuming (x,y) are **pixels**.
  - width/height are **microns** and are internally converted using micronsPerPixel.
  - "filename" is shown from spectraplots/<filename> on SHIFT-hover.
  - "intensity" is rendered as opacity (0..1).

Example:
  python create_eds_grid.py /path/to/folder \
    --make-plots --intensity total --clip-percent 1.0 --eds-dir /path/to/h5data
"""

import os, sys, csv, json, math, argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# --------------------------- CLI ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Create rectangles + intensity overlay JSON for EDS viewer.")
    p.add_argument("folder", help="Folder containing summary_table.csv")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: same as folder)")
    p.add_argument("--rectangles-name", default="rectangles.json",
                   help="Filename for rectangles JSON (micron units)")
    p.add_argument("--intensity-json", default="adjusted_intensity_rectangles2.json",
                   help="Filename for intensity overlay JSON")
    p.add_argument("--plots-dir", default="spectraplots",
                   help="Subfolder to write per-tile spectrum PNGs")
    p.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Write per-tile spectrum PNGs (default: true)")
    p.add_argument("--skip-existing-plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip writing a plot if file already exists (default: true)")
    p.add_argument("--eds-dir", default=None,
                   help="Where to search for <basename>_eds.npz if not in same dir")
    p.add_argument("--eds-key", default=None,
                   help="NPZ key for spectra (default: auto; tries 'eds_data' then first array)")
    p.add_argument("--intensity", choices=["total", "window", "mean", "max"], default="total",
                   help="Intensity metric derived from the averaged spectrum")
    p.add_argument("--window-kev", default=None,
                   help="If --intensity=window, energy window in keV as 'lo,hi' (e.g., '0.5,10')")
    p.add_argument("--clip-percent", type=float, default=1.0,
                   help="Percentile clip for intensity normalization (default 1.0)")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                   help="Print extra info (default: true)")
    return p.parse_args()


# --------------------------- I/O helpers -------------------------------------

def read_summary_csv(csv_path: Path) -> List[Dict[str, Any]]:
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[error] No rows in {csv_path}")
    need = ["basename", "width_px", "height_px", "px_x_um", "px_y_um", "X_rel_um", "Y_rel_um"]
    for k in need:
        if k not in rows[0]:
            raise SystemExit(f"[error] Column '{k}' missing in {csv_path}")
    return rows

def parse_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def parse_int(x, default=None):
    try:
        return int(round(float(x)))
    except Exception:
        return default


# --------------------------- EDS loading + metrics ---------------------------

def find_eds_npz(row: Dict[str, Any], eds_dir: Optional[Path]) -> Optional[Path]:
    """
    Resolve EDS NPZ path for a tile:
      1) If 'eds_npz_path' column present and file exists -> use it.
      2) Else guess from basename: "<basename>_eds.npz" in:
          - same folder as summary CSV, or
          - provided --eds-dir (if given).
    """
    cand = row.get("eds_npz_path")
    if cand and Path(cand).is_file():
        return Path(cand)

    base = (row.get("basename") or "").strip()
    if not base:
        return None
    guess_name = base + "_eds.npz"

    # Try png/npz directory first, then eds_dir if given
    # Look near png_path/npz_path if present
    for col in ("npz_path", "png_path"):
        p = row.get(col)
        if p:
            here = Path(p).parent
            guess = here / guess_name
            if guess.is_file():
                return guess

    if eds_dir:
        guess = Path(eds_dir) / guess_name
        if guess.is_file():
            return guess

    return None


def average_spectrum_from_npz(npz_path: Path, key_hint: Optional[str] = None) -> np.ndarray:
    """
    Load EDS spectra NPZ and return an averaged spectrum 1D array (length C).
    Accepts either 'eds_data' or the first array in the archive.
    'eds_data' expected shape: (H, W, C) or (Npix, C).
    """
    d = np.load(npz_path)
    if key_hint and key_hint in d:
        a = d[key_hint]
    elif "eds_data" in d:
        a = d["eds_data"]
    else:
        first = d.files[0]
        a = d[first]

    a = np.asarray(a)
    if a.ndim == 3:
        # H,W,C -> N,C
        H, W, C = a.shape
        spec = a.reshape(-1, C).mean(axis=0)
    elif a.ndim == 2:
        # N,C -> mean over N
        spec = a.mean(axis=0)
    elif a.ndim == 1:
        # already a spectrum
        spec = a
    else:
        raise ValueError(f"Unsupported EDS array shape {a.shape} in {npz_path}")
    return np.asarray(spec, dtype=np.float64)


def intensity_from_spectrum(spec: np.ndarray,
                            mode: str = "total",
                            window_kev: Optional[str] = None,
                            meta: Optional[Dict[str, Any]] = None) -> float:
    """
    Compute intensity metric from a (mean) spectrum.
    - 'total': sum over all channels
    - 'mean':  mean over all channels
    - 'max':   max across channels
    - 'window': sum over channels in [lo_keV, hi_keV].
        Requires knowledge of channel -> energy mapping.
        If metadata is unavailable, assumes 0..(C-1) are equal steps across 0..20 keV.
    """
    mode = (mode or "total").lower()
    if mode == "total":
        return float(np.nansum(spec))
    if mode == "mean":
        return float(np.nanmean(spec))
    if mode == "max":
        return float(np.nanmax(spec))
    if mode == "window":
        if not window_kev:
            # fallback to total if no window provided
            return float(np.nansum(spec))

        try:
            lo_keV, hi_keV = [float(x) for x in window_kev.split(",")]
        except Exception:
            return float(np.nansum(spec))

        C = spec.shape[0]

        # Try to derive mapping from metadata if available
        # Expected (if provided): start_eV, width_eV, number_channels
        start_eV = parse_float((meta or {}).get("start_eV"), None)
        width_eV = parse_float((meta or {}).get("width_eV"), None)
        # fallback: 0..20 keV evenly spread
        if start_eV is None or width_eV is None:
            e_keV = np.linspace(0.0, 20.0, C, dtype=np.float64)
        else:
            # start_eV may be negative; width in eV/channel
            e_keV = (start_eV + np.arange(C) * width_eV) / 1000.0

        lo_idx = np.searchsorted(e_keV, lo_keV, side="left")
        hi_idx = np.searchsorted(e_keV, hi_keV, side="right")
        lo_idx = max(0, min(C, lo_idx))
        hi_idx = max(0, min(C, hi_idx))
        if hi_idx <= lo_idx:
            return 0.0
        return float(np.nansum(spec[lo_idx:hi_idx]))

    # default
    return float(np.nansum(spec))


def save_spectrum_plot(spec: np.ndarray, out_png: Path, title: Optional[str] = None):
    if not HAVE_MPL:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(spec.size), spec, lw=1.0)
    plt.title(title or out_png.stem)
    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


# --------------------------- geometry helpers --------------------------------

def compute_pixel_xy(row: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Convert relative stage microns to pixel offsets:
      x_px = X_rel_um / px_x_um
      y_px = Y_rel_um / px_y_um
    Apply invertx/inverty if present (1 means invert).
    """
    x_um = parse_float(row.get("X_rel_um"), None)
    y_um = parse_float(row.get("Y_rel_um"), None)
    px_x_um = parse_float(row.get("px_x_um"), None)
    px_y_um = parse_float(row.get("px_y_um"), None)
    if None in (x_um, y_um, px_x_um, px_y_um) or px_x_um == 0 or px_y_um == 0:
        return None

    invx = parse_int(row.get("invertx", 0), 0)
    invy = parse_int(row.get("inverty", 0), 0)
    if invx == 1:
        x_um = -x_um
    if invy == 1:
        y_um = -y_um

    x_px = x_um / px_x_um
    y_px = y_um / px_y_um
    return {"x_px": float(np.round(x_px)), "y_px": float(np.round(y_px))}


def micron_size(row: Dict[str, Any]) -> Optional[Dict[str, float]]:
    wpx = parse_int(row.get("width_px"), None)
    hpx = parse_int(row.get("height_px"), None)
    px_x_um = parse_float(row.get("px_x_um"), None)
    px_y_um = parse_float(row.get("px_y_um"), None)
    if None in (wpx, hpx, px_x_um, px_y_um):
        return None
    return {"w_um": float(wpx * px_x_um), "h_um": float(hpx * px_y_um)}


# --------------------------- main --------------------------------------------

def main():
    args = parse_args()
    folder = Path(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = folder / "summary_table.csv"
    rows = read_summary_csv(csv_path)

    # --- 1) rectangles.json (micron units, legacy) ---------------------------
    rects_micron = []
    for r in rows:
        name = (r.get("basename") or "").strip() or (Path(r.get("png_path","")).stem or "tile")
        size_um = micron_size(r)
        if size_um is None:
            continue
        x_um = parse_float(r.get("X_rel_um"), None)
        y_um = parse_float(r.get("Y_rel_um"), None)
        if x_um is None or y_um is None:
            continue
        # respect invert flags for legacy rectangles too
        invx = parse_int(r.get("invertx", 0), 0)
        invy = parse_int(r.get("inverty", 0), 0)
        if invx == 1: x_um = -x_um
        if invy == 1: y_um = -y_um

        rects_micron.append({
            "id": name,
            "name": name,
            "x": float(x_um),
            "y": float(y_um),
            "width": size_um["w_um"],
            "height": size_um["h_um"],
        })

    rects_path = out_dir / args.rectangles_name
    with open(rects_path, "w", encoding="utf-8") as f:
        json.dump(rects_micron, f, indent=2)
    if args.verbose:
        print(f"wrote {rects_path}  ({len(rects_micron)} rectangles, microns)")

    # --- 2) adjusted_intensity_rectangles2.json ------------------------------
    intensity_items = []
    intensities = []

    plots_dir = out_dir / args.plots_dir
    if args.make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for r in rows:
        # geometry
        pix = compute_pixel_xy(r)
        size_um = micron_size(r)
        if (pix is None) or (size_um is None):
            continue

        basename = (r.get("basename") or "").strip()
        display_name = basename if basename else (Path(r.get("png_path","")).stem or "tile")
        plot_filename = display_name + ".png"

        # EDS path
        eds_path = find_eds_npz(r, Path(args.eds_dir) if args.eds_dir else None)

        # default: missing EDS => intensity 0, optional blank plot
        intensity_val = 0.0
        if eds_path and eds_path.is_file():
            try:
                spec = average_spectrum_from_npz(eds_path, key_hint=args.eds_key)
                # If you have per-file metadata to map channels -> keV precisely,
                # pass in dict like {"start_eV":..., "width_eV":...}
                intensity_val = intensity_from_spectrum(
                    spec,
                    mode=args.intensity,
                    window_kev=args.window_kev,
                    meta=None  # fill with metadata if available
                )
                if args.make_plots:
                    out_png = plots_dir / plot_filename
                    if (not args.skip_existing_plots) or (not out_png.exists()):
                        save_spectrum_plot(spec, out_png, title=display_name)
            except Exception as e:
                if args.verbose:
                    print(f"[warn] EDS read failed for {eds_path.name}: {e}")
        else:
            if args.verbose:
                print(f"[warn] EDS file not found for '{display_name}'")

        intensities.append(intensity_val)
        intensity_items.append({
            # viewer expects x,y in **pixels**
            "x": pix["x_px"],
            "y": pix["y_px"],
            # width/height in **microns**
            "width": size_um["w_um"],
            "height": size_um["h_um"],
            # tooltip image filename (viewer prefixes spectraplots/)
            "filename": plot_filename,
            # temp raw intensity (normalize later)
            "_raw_intensity": intensity_val,
        })

    # Normalize intensity -> [0,1] with optional percentile clipping
    if intensity_items:
        arr = np.asarray(intensities, dtype=np.float64)
        p = float(args.clip_percent or 0.0)
        if p > 0.0 and np.isfinite(arr).any():
            lo = np.nanpercentile(arr, p)
            hi = np.nanpercentile(arr, 100.0 - p)
        else:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 1.0
        if hi <= lo:
            # degenerate, map all to 0.5
            for item in intensity_items:
                item["intensity"] = 0.5
        else:
            for item in intensity_items:
                v = float(item.pop("_raw_intensity", 0.0))
                vv = (v - lo) / (hi - lo)
                item["intensity"] = float(np.clip(vv, 0.0, 1.0))

    out_intensity = out_dir / args.intensity_json
    with open(out_intensity, "w", encoding="utf-8") as f:
        json.dump(intensity_items, f, indent=2)
    if args.verbose:
        print(f"wrote {out_intensity}  ({len(intensity_items)} tiles with intensity)")

    print("All done.")


if __name__ == "__main__":
    main()
