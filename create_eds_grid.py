#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a rectangles grid JSON for the next processing step, using summary_table.csv.

Inputs (required columns in summary_table.csv):
  npz_path, width_px, height_px, px_x_um, px_y_um, X_rel_um, Y_rel_um
Optional columns:
  invertx, inverty  (1 means flip sign of that axis)

The output rectangles are in microns, with the stage origin rebased so that min(x,y) -> (0,0).

Usage examples
--------------
# simplest: write rectangles.json next to the CSV
python create_grid_from_summary.py "/path/to/pythondata"

# apply small empirical tweaks if you need them
python create_grid_from_summary.py "/path/to/pythondata" \
  --scale-x 1.000 --scale-y 1.000 --offset-x-um 0 --offset-y-um 0
"""

from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path
from typing import List, Dict, Any

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate rectangles.json from summary_table.csv"
    )
    p.add_argument(
        "folder",
        help="Folder that contains summary_table.csv"
    )
    p.add_argument(
        "--out", default=None,
        help="Output JSON path (default: <folder>/rectangles.json)"
    )
    # Optional gentle tweaks (replaces hard-coded fudge factors that were in the notebook)
    p.add_argument("--scale-x", type=float, default=1.0, help="Global X scale multiplier")
    p.add_argument("--scale-y", type=float, default=1.0, help="Global Y scale multiplier")
    p.add_argument("--offset-x-um", type=float, default=0.0, help="Global X offset (µm) after rebase")
    p.add_argument("--offset-y-um", type=float, default=0.0, help="Global Y offset (µm) after rebase")
    p.add_argument("--preview-csv", action=argparse.BooleanOptionalAction, default=True,
                   help="Also write a rectangles_preview.csv for quick inspection (default true)")
    return p.parse_args()

def read_summary_csv(csv_path: Path) -> List[Dict[str, Any]]:
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise SystemExit(f"[error] No rows found in {csv_path}")
    need = ["npz_path","width_px","height_px","px_x_um","px_y_um","X_rel_um","Y_rel_um"]
    for k in need:
        if k not in rows[0]:
            raise SystemExit(f"[error] '{k}' column missing in {csv_path}")
    return rows

def parse_int(x, default=0):
    try:
        return int(round(float(x)))
    except Exception:
        return default

def parse_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def main():
    args = parse_args()
    folder = Path(args.folder)
    csv_path = folder / "summary_table.csv"
    if not csv_path.exists():
        raise SystemExit(f"[error] {csv_path} not found")

    rows = read_summary_csv(csv_path)

    # Build raw rectangles in *microns*, using stage positions and per-tile pixel size
    rects = []
    has_invx = "invertx" in rows[0]
    has_invy = "inverty" in rows[0]

    for i, r in enumerate(rows):
        # stage positions in microns (relative)
        x_um = parse_float(r["X_rel_um"], 0.0)
        y_um = parse_float(r["Y_rel_um"], 0.0)

        if has_invx and parse_int(r.get("invertx", 0)) == 1:
            x_um = -x_um
        if has_invy and parse_int(r.get("inverty", 0)) == 1:
            y_um = -y_um

        # tile size in microns
        w_px = parse_int(r["width_px"], 0)
        h_px = parse_int(r["height_px"], 0)
        px_x_um = parse_float(r["px_x_um"], 0.0)
        px_y_um = parse_float(r["px_y_um"], 0.0)
        if px_x_um <= 0 or px_y_um <= 0:
            raise SystemExit("[error] Non-positive px_x_um/px_y_um found")

        width_um  = w_px * px_x_um
        height_um = h_px * px_y_um

        rects.append({
            "id": i,
            "name": Path(r.get("npz_path","")).name or f"tile_{i}",
            "x_um": x_um,
            "y_um": y_um,
            "width_um": width_um,
            "height_um": height_um,
        })

    if not rects:
        raise SystemExit("[error] No rectangles built")

    # Rebase so minimum XY is (0,0)
    min_x = min(r["x_um"] for r in rects)
    min_y = min(r["y_um"] for r in rects)
    for r in rects:
        r["x_um"] = r["x_um"] - min_x
        r["y_um"] = r["y_um"] - min_y

    # Apply global tweak (if user wants to mirror previous notebook behavior)
    sx, sy = float(args.scale_x), float(args.scale_y)
    ox, oy = float(args.offset_x_um), float(args.offset_y_um)
    for r in rects:
        r["x_um"] = r["x_um"] * sx + ox
        r["y_um"] = r["y_um"] * sy + oy
        r["width_um"]  = r["width_um"]  * sx
        r["height_um"] = r["height_um"] * sy

    # Final JSON schema the next step expects: x, y, width, height (all µm) + id/name
    out_rects = [
        {
            "id": r["id"],
            "name": r["name"],
            "x": r["x_um"],
            "y": r["y_um"],
            "width": r["width_um"],
            "height": r["height_um"],
        }
        for r in rects
    ]

    out_json = Path(args.out) if args.out else (folder / "rectangles.json")
    with open(out_json, "w") as f:
        json.dump(out_rects, f, indent=2)
    print(f"✅ wrote {out_json}")

    if args.preview_csv:
        prev = out_json.with_name(out_json.stem.replace(".json","") + "_preview.csv") \
               if out_json.suffix == ".json" else out_json.with_name("rectangles_preview.csv")
        with open(prev, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","name","x_um","y_um","width_um","height_um"])
            for r in out_rects:
                w.writerow([r["id"], r["name"], f"{r['x']:.3f}", f"{r['y']:.3f}",
                            f"{r['width']:.3f}", f"{r['height']:.3f}"])
        print(f"📝 wrote {prev}")

if __name__ == "__main__":
    main()
