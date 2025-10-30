#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Populate adjusted_intensity_rectangles2.json from precomputed window TXT files.
- geometry: processed_dir/rectangles.json (already produced earlier)
- intensities: windows_dir/*channel_*_to_*.txt (columns include frac_window_of_total)
- tooltip image names: processed_dir/<plots-dir>/...

Writes: processed_dir/<rectangles-name>.json
"""

import os, csv, json, argparse, re
from pathlib import Path
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Fill intensities for rectangles from precomputed TXT window files.")
    p.add_argument("processed_dir",
                   help="Folder that contains rectangles.json (and usually summary_table.csv)")
    p.add_argument("--windows-dir", default=None,
                   help="Folder containing *channel_*_to_*.txt files (default: processed_dir)")
    p.add_argument("--window-glob", default="*channel_*_to_*.txt",
                   help="Glob for the intensity TXT files (default: *channel_*_to_*.txt)")
    p.add_argument("--rectangles-json", default="rectangles.json",
                   help="Input rectangles JSON (default: rectangles.json)")
    p.add_argument("--rectangles-name", default="adjusted_intensity_rectangles2",
                   help="Output basename (default: adjusted_intensity_rectangles2)")
    p.add_argument("--plots-dir", default="spectraplots",
                   help="Directory with preview PNGs used in tooltips (default: spectraplots)")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()

def normalize_base(name):
    """Return a clean base string from id/name; tolerate non-strings."""
    if name is None:
        return ""
    s = str(name)
    s = re.sub(r"\.(npz|npy|png|tif|tiff)$", "", s, flags=re.I)
    s = re.sub(r"_(eds|sem|all-segments)$", "", s, flags=re.I)
    return s

def read_intensity_txt(txt_path: Path):
    """
    Expect lines like:
      base, stage_x_um, stage_y_um, row, col, frac_window_of_total
    Header lines starting with '#' are ignored.
    Returns: list[(base,row,col,intensity)]
    """
    out = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            # tolerate extra columns; require at least 6
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

def load_all_intensities(windows_dir: Path, pattern: str, verbose: bool):
    table = defaultdict(dict)  # base -> {(row,col): intensity}
    files = sorted(windows_dir.glob(pattern))
    if verbose:
        print(f"[info] scanning {windows_dir} for '{pattern}' -> {len(files)} files")
    for p in files:
        vals = read_intensity_txt(p)
        for base, r, c, inten in vals:
            table[base][(r, c)] = inten
    if verbose:
        print(f"[info] collected intensities for {len(table)} tiles (base names)")
    return table

def decide_png_name(plots_dir: Path, base: str, r: int, c: int):
    seg = plots_dir / f"{base}_r{r}_c{c}.png"
    if seg.exists():
        return seg.name
    whole = plots_dir / f"{base}.png"
    if whole.exists():
        return whole.name
    return f"{base}.png"

def coerce_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def main():
    args = parse_args()
    proc = Path(args.processed_dir)
    windows_dir = Path(args.windows_dir) if args.windows_dir else proc
    rect_json_in = proc / args.rectangles_json
    out_path = proc / f"{args.rectangles_name}.json"
    plots_dir = proc / args.plots_dir

    if not rect_json_in.exists():
        raise SystemExit(f"[error] missing {rect_json_in}")

    rectangles = json.loads(rect_json_in.read_text())

    intens_lookup = load_all_intensities(windows_dir, args.window_glob, args.verbose)
    if args.verbose and not intens_lookup:
        print("[warn] no intensity TXT files were found — intensities will be 0.0")

    filled, missing = 0, 0
    for rec in rectangles:
        # base id/name
        base = normalize_base(
            rec.get("id") or rec.get("name") or rec.get("base")
        )

        # rows/cols – be forgiving about key names
        r = coerce_int(rec.get("row", rec.get("r")))
        c = coerce_int(rec.get("col", rec.get("c")))

        inten = None
        tile_map = intens_lookup.get(base)
        if tile_map:
            if r is not None and c is not None:
                inten = tile_map.get((r, c))
            if inten is None and len(tile_map) == 1:
                # single-segment case
                inten = next(iter(tile_map.values()))

        if inten is None:
            missing += 1
            rec["intensity"] = 0.0
        else:
            rec["intensity"] = float(inten)
            filled += 1

        rr = r if r is not None else 0
        cc = c if c is not None else 0
        rec["filename"] = decide_png_name(plots_dir, base, rr, cc)

    out_path.write_text(json.dumps(rectangles, indent=2))
    print(f"wrote {out_path}  ({len(rectangles)} rectangles)  filled={filled}, missing={missing}")

if __name__ == "__main__":
    main()
