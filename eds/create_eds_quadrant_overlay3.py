#!/usr/bin/env python3
import os, json, argparse, math
from glob import glob

def load_tile_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def window_sum(spectrum, start, end):
    n = len(spectrum)
    s = max(0, min(start, n - 1))
    e = max(0, min(end,   n - 1))
    if e < s:
        s, e = e, s
    return float(sum(spectrum[s:e+1]))

def robust_min_max(values, lo_pct=1.0, hi_pct=99.0):
    if not values: return (0.0, 1.0)
    v = sorted(values)
    def p(q):
        i = (len(v) - 1) * q / 100.0
        lo, hi = math.floor(i), math.ceil(i)
        return v[int(i)] if lo == hi else v[lo] + (v[hi] - v[lo]) * (i - lo)
    lo, hi = p(lo_pct), p(hi_pct)
    if hi <= lo:
        lo, hi = v[0], v[-1]
        if hi <= lo: hi = lo + 1.0
    return lo, hi

def main():
    ap = argparse.ArgumentParser(
        description="Build microns-based quadrant heatmap rectangles from per-tile JSONs (with srcJson for spectra tooltips)."
    )
    ap.add_argument("input_folder", help="Folder containing per-tile JSON files (e.g. aggregated-spectra_3x3_json)")
    ap.add_argument("window_start", type=int, help="Start channel (inclusive)")
    ap.add_argument("window_end",   type=int, help="End channel (inclusive)")
    ap.add_argument("-o","--output", help="Output JSON (default: window_<start>_<end>_heatmap.json in input_folder)")
    ap.add_argument("--gamma", type=float, default=0.6, help="Gamma for opacity contrast (default 0.6)")
    ap.add_argument("--robust", action="store_true", help="Use robust 1–99th percentile for min/max")
    ap.add_argument("--fill", default="#ff0000", help="Fill color (default red)")
    ap.add_argument("--stroke-px", type=float, default=1.5, help="Outline width in on-screen pixels (default 1.5)")
    args = ap.parse_args()

    in_dir  = os.path.abspath(args.input_folder)
    out_path = args.output or os.path.join(in_dir, f"window_{args.window_start}_{args.window_end}_heatmap.json")

    # All per-tile JSONs (ignore the would-be output if it’s in same folder)
    files = sorted(
        p for p in glob(os.path.join(in_dir, "*.json"))
        if os.path.abspath(p) != os.path.abspath(out_path)
    )
    if not files:
        raise SystemExit(f"No JSON files found in {in_dir}")

    rectangles, sums = [], []
    folder_name = os.path.basename(os.path.normpath(in_dir))  # becomes part of srcJson

    for jf in files:
        try:
            entries = load_tile_entries(jf)
        except Exception:
            continue
        if not isinstance(entries, list) or not entries:
            continue

        # srcJson must be relative to dataset base path (viewer’s `base`)
        # Example: "aggregated-spectra_3x3_json/Project ... 12.json"
        src_json_rel = f"{folder_name}/{os.path.basename(jf)}"

        # infer grid (e.g., 3x3)
        max_col = max(int(e["colnum"]) for e in entries)
        max_row = max(int(e["rownum"]) for e in entries)
        cols, rows = max_col + 1, max_row + 1

        for e in entries:
            base_x  = float(e["X_rel_um"])
            base_y  = float(e["Y_rel_um"])
            tile_w  = float(e["TileWidth_um"])
            tile_h  = float(e["TileHeight_um"])
            c       = int(e["colnum"])
            r       = int(e["rownum"])
            spec    = e["aggregatedspectrum"]

            # explicit quadrant geometry if present (recommended)
            qw = float(e.get("quad_width_um",  tile_w / cols))
            qh = float(e.get("quad_height_um", tile_h / rows))
            qx = base_x + float(e.get("quad_x_um", c * (tile_w / cols)))
            qy = base_y + float(e.get("quad_y_um", r * (tile_h / rows)))

            s = window_sum(spec, args.window_start, args.window_end)
            sums.append(s)

            rectangles.append({
                "type": "rect",
                "units": "microns",
                "x": qx, "y": qy, "width": qw, "height": qh,
                "fill": args.fill,
                "stroke": "#000000",
                "strokeWidthPx": args.stroke_px,   # viewer uses this for crisp borders
                "strokeWidth": 0.0005,            # harmless; kept for back-compat
                "intensity_raw": s,               # will be normalized below
                "basename": e.get("basename",""),
                "colnum": c,
                "rownum": r,
                "srcJson": src_json_rel,          # <-- key for spectrum tooltip loader
                # Optional label shown in tooltip
                "label": f'{e.get("basename","")} · r{r}, c{c}'
            })

    # Normalize to [0,1] (optionally robust), gamma map → fillOpacity
    if sums:
        if args.robust:
            lo, hi = robust_min_max(sums, 1.0, 99.0)
        else:
            lo, hi = min(sums), max(sums)
            if hi <= lo: hi = lo + 1.0
        rng = hi - lo
        for rect in rectangles:
            v = (rect.pop("intensity_raw") - lo) / rng
            v = max(0.0, min(1.0, v))
            if args.gamma and args.gamma > 0:
                v = v ** args.gamma
            rect["fillOpacity"] = round(v, 6)
            rect["intensity"]   = round(v, 6)
    else:
        for rect in rectangles:
            rect["fillOpacity"] = 0.0
            rect["intensity"]   = 0.0
            rect.pop("intensity_raw", None)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rectangles, f, indent=2)

    print(f"Wrote {len(rectangles)} rectangles → {out_path}")

if __name__ == "__main__":
    main()

