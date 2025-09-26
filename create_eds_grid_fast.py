#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight: fill intensities into rectangles from precomputed window files.
- geometry from summary_table.csv
- intensities from *_channel_*.txt  (columns: base,row,col,frac_window_of_total)
- tooltip image from spectraplots/<base>.png (or <base>_r{row}_c{col}.png if present)
Writes adjusted_intensity_rectangles2.json (same schema the viewer uses).
"""

import os, csv, json, argparse, re
from pathlib import Path
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Populate adjusted_intensity_rectangles2.json from precomputed intensity txt files.")
    p.add_argument("processed_dir", help="Folder that contains summary_table.csv and rectangles.json")
    p.add_argument("--window-glob", default="*channel_*_to_*.txt",
                   help="Glob for precomputed intensity files (default: *channel_*_to_*.txt)")
    p.add_argument("--rectangles-name", default="adjusted_intensity_rectangles2",
                   help="Base name for output JSON (default: adjusted_intensity_rectangles2)")
    p.add_argument("--plots-dir", default="spectraplots",
                   help="Directory with preview PNGs used in tooltips (default: spectraplots)")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()

def load_summary_table(csv_path: Path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise SystemExit(f"[error] empty {csv_path}")
    # we mainly need a lookup from base name -> (width_um,height_um, px_um, offsets) if ever needed
    return rows

def normalize_base(name: str):
    # drop common suffixes (.npz, _eds, _sem, etc.)
    n = re.sub(r"\.(npz|npy|png|tif|tiff)$", "", name)
    n = re.sub(r"_(eds|sem|all-segments)$", "", n, flags=re.I)
    return n

def read_intensity_txt(txt_path: Path):
    """
    Expect lines like:
      base, stage_x_um, stage_y_um, row, col, frac_window_of_total
    Header lines starting with '#' are ignored.
    Returns: list of (base,row,col,intensity)
    """
    out = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 6:
                continue
            base = normalize_base(parts[0])
            try:
                row = int(parts[3]); col = int(parts[4])
                inten = float(parts[5])
            except Exception:
                continue
            out.append((base, row, col, inten))
    return out

def load_all_intensities(intensity_dir: Path, pattern: str, verbose: bool):
    # dict: base -> dict[(row,col)] = intensity
    table = defaultdict(dict)
    files = sorted(intensity_dir.glob(pattern))
    if verbose:
        print(f"[info] scanning {intensity_dir} for '{pattern}' -> {len(files)} files")
    for p in files:
        vals = read_intensity_txt(p)
        for base, r, c, inten in vals:
            table[base][(r, c)] = inten
    if verbose:
        print(f"[info] collected intensities for {len(table)} tiles")
    return table

def decide_png_name(plots_dir: Path, base: str, r: int, c: int):
    # Prefer per-segment PNG if present, otherwise per-tile PNG
    seg = plots_dir / f"{base}_r{r}_c{c}.png"
    if seg.exists():
        return seg.name
    whole = plots_dir / f"{base}.png"
    if whole.exists():
        return whole.name
    # fallback: nothing
    return f"{base}.png"

def main():
    args = parse_args()
    proc = Path(args.processed_dir)
    summary_csv = proc / "summary_table.csv"
    rect_json_in = proc / "rectangles.json"  # produced earlier by your create_eds_grid step
    plots_dir = proc / args.plots_dir

    if not rect_json_in.exists():
        raise SystemExit(f"[error] missing {rect_json_in}")
    if not summary_csv.exists():
        print(f"[warn] {summary_csv} not found — continuing (not strictly needed for this step).")

    # Load rectangles.json (must already contain x,y,width,height and id/name and row/col if available)
    rectangles = json.loads(rect_json_in.read_text())

    # Build intensity lookup from precomputed txt files
    intens = load_all_intensities(proc, args.window_glob, args.verbose)

    # Fill in intensity and filename
    filled, missing = 0, 0
    for rec in rectangles:
        base = normalize_base(rec.get("id") or rec.get("name") or "")
        r = rec.get("row")
        c = rec.get("col")
        inten = None
        if base in intens:
            if r is not None and c is not None:
                inten = intens[base].get((int(r), int(c)))
            # If row/col not present in rectangles, try single value (e.g., only one segment)
            if inten is None and len(intens[base]) == 1:
                inten = next(iter(intens[base].values()))
        if inten is None:
            missing += 1
            rec["intensity"] = 0.0
        else:
            rec["intensity"] = float(inten)
            filled += 1

        # Tooltip image filename
        rr = int(rec.get("row")) if rec.get("row") is not None else 0
        cc = int(rec.get("col")) if rec.get("col") is not None else 0
        rec["filename"] = decide_png_name(plots_dir, base, rr, cc)

    out_path = proc / f"{args.rectangles_name}.json"
    out_path.write_text(json.dumps(rectangles, indent=2))
    print(f"wrote {out_path}  ({len(rectangles)} rectangles)  filled={filled}, missing={missing}")

if __name__ == "__main__":
    main()
