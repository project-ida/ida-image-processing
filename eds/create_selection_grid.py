#!/usr/bin/env python3
"""
create_selection_grid.py

Make a *selection* overlay (no heatmap) from the same per-tile JSONs you used
for the heatmap script. The output is a JSON array of rectangles in **microns**
with all the metadata the viewer expects:

- type: "rect"
- units: "microns"
- x, y, width, height
- srcJson   (relative path to the per-tile json, e.g. "aggregated-spectra_3x3_json/file.json")
- rownum / colnum
- basename
- label

Differences to the heatmap script:
- NO intensity calculation
- fill is transparent (or almost)
- stroke is visible so rectangles can be selected
- optional shrinking of each quadrant to avoid overlapping strokes
"""

import os
import json
import argparse
from glob import glob


def load_tile_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Create a selection grid (transparent rects with outlines) from quadrant JSONs."
    )
    ap.add_argument(
        "input_folder",
        help="Folder containing per-tile JSON files (e.g. aggregated-spectra_3x3_json/)"
    )
    ap.add_argument(
        "-o", "--output",
        help="Output JSON (default: selection_grid.json in input_folder)"
    )
    ap.add_argument(
        "--stroke", default="#000000",
        help="Outline color for each quadrant (default: black)"
    )
    ap.add_argument(
        "--stroke-px", type=float, default=1.0,
        help="On-screen stroke width in pixels (default: 1.0)"
    )
    ap.add_argument(
        "--fill", default="none",
        help="Fill color (default: 'none'); can be rgba(...) if you want a slight tint"
    )
    ap.add_argument(
        "--fill-opacity", type=float, default=0.0,
        help="Fill opacity 0..1 (default: 0.0 → transparent)"
    )
    ap.add_argument(
        "--shrink-pct", type=float, default=0.0,
        help="Shrink each rect by this percent on EACH side in microns space (default 0.0). "
             "E.g. 3.0 → shrink width/height by 3%% on left+right and 3%% on top+bottom."
    )

    args = ap.parse_args()

    in_dir = os.path.abspath(args.input_folder)

    if args.output:
        # User explicitly set an output path
        out_path = args.output
    else:
        # Default behaviour:
        #   ../overlays/selection_grid.json
        parent = os.path.dirname(in_dir)
        overlay_dir = os.path.join(parent, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        out_path = os.path.join(overlay_dir, "selection_grid.json")

    files = sorted(glob(os.path.join(in_dir, "*.json")))
    if not files:
        raise SystemExit(f"No JSON files found in {in_dir}")

    folder_name = os.path.basename(os.path.normpath(in_dir))

    rectangles = []

    for jf in files:
        try:
            entries = load_tile_entries(jf)
        except Exception:
            continue

        if not isinstance(entries, list) or not entries:
            continue

        # make relative srcJson as the viewer expects
        src_json_rel = f"{folder_name}/{os.path.basename(jf)}"

        # we don’t actually *need* cols/rows here except for fallback geometry
        max_col = max(int(e["colnum"]) for e in entries)
        max_row = max(int(e["rownum"]) for e in entries)
        cols, rows = max_col + 1, max_row + 1

        for e in entries:
            base_x = float(e["X_rel_um"])
            base_y = float(e["Y_rel_um"])
            tile_w = float(e["TileWidth_um"])
            tile_h = float(e["TileHeight_um"])
            c = int(e["colnum"])
            r = int(e["rownum"])

            # preferred explicit quadrant geometry if present
            qw = float(e.get("quad_width_um", tile_w / cols))
            qh = float(e.get("quad_height_um", tile_h / rows))
            qx = base_x + float(e.get("quad_x_um", c * (tile_w / cols)))
            qy = base_y + float(e.get("quad_y_um", r * (tile_h / rows)))

            # optional shrinking — we stay in microns
            if args.shrink_pct > 0:
                shrink_x = qw * (args.shrink_pct / 100.0)
                shrink_y = qh * (args.shrink_pct / 100.0)
                qx += shrink_x
                qy += shrink_y
                qw -= 2 * shrink_x
                qh -= 2 * shrink_y
                if qw < 0:
                    qw = 0
                if qh < 0:
                    qh = 0

            rect = {
                "type": "rect",
                "units": "microns",
                "x": qx,
                "y": qy,
                "width": qw,
                "height": qh,
                "fill": args.fill,
                "fillOpacity": args.fill_opacity,
                "stroke": args.stroke,
                "strokeWidthPx": args.stroke_px,
                # keep the viewer metadata:
                "basename": e.get("basename", ""),
                "colnum": c,
                "rownum": r,
                "srcJson": src_json_rel,
                "label": f'{e.get("basename","")} · r{r}, c{c}',
            }

            rectangles.append(rect)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rectangles, f, indent=2)

    print(f"Wrote {len(rectangles)} selection rectangles → {out_path}")


if __name__ == "__main__":
    main()
