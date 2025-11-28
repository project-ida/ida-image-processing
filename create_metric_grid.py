#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# summary_table.csv -> stitched area size (mm)
# Assumes header:
# Index,PosIndex,Z_layer,Z_offset_um,Z_base_um,Z_um,Label,GridRow,GridCol,
# X_um,Y_um,X_rel_um,Y_rel_um,Z_rel_um,TileWidth_um,TileHeight_um,
# TileWidth_px,TileHeight_px,PixelSizeX_um,PixelSizeY_um
# ---------------------------------------------------------------------------

def dims_from_summary(folder: Path) -> Tuple[float, float]:
    csv_path = folder / "summary_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_um = float(row["X_rel_um"])
            y_um = float(row["Y_rel_um"])
            w_um = float(row["TileWidth_um"])
            h_um = float(row["TileHeight_um"])

            min_x = min(min_x, x_um)
            min_y = min(min_y, y_um)
            max_x = max(max_x, x_um + w_um)
            max_y = max(max_y, y_um + h_um)

    if not math.isfinite(min_x) or not math.isfinite(max_x) or not math.isfinite(min_y) or not math.isfinite(max_y):
        raise ValueError("Could not determine stitched extents from summary_table.csv.")

    width_um  = max(0.0, max_x - min_x)
    height_um = max(0.0, max_y - min_y)
    return width_um / 1000.0, height_um / 1000.0  # -> mm


# ---------------------------------------------------------------------------
# grid/labels generator (in microns)
# ---------------------------------------------------------------------------

def index_to_col(idx: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA', ..."""
    if idx < 0:
        raise ValueError("index must be >= 0")
    out = ""
    n = idx
    while True:
        n, rem = divmod(n, 26)
        out = chr(65 + rem) + out
        if n == 0:
            break
        n -= 1
    return out


def build_metric_grid(
    width_mm: float,
    height_mm: float,
    cell_mm: float = 1.0,
    font_mm: float = 0.15,
):
    """
    Returns (grid_items, label_items) where both are microns-based dicts.
    Rects are 'type':'rect'; labels are 'type':'text'.
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("width_mm and height_mm must be > 0")

    n_x = int(math.floor(width_mm / cell_mm))
    n_y = int(math.floor(height_mm / cell_mm))

    grid_items = []
    label_items = []

    font_um = font_mm * 1000.0
    cell_um = cell_mm * 1000.0

    for ix in range(n_x):
        for iy in range(n_y):
            x_um = ix * cell_um
            y_um = iy * cell_um

            # grid cell
            grid_items.append({
                "type": "rect",
                "units": "microns",
                "x": x_um,
                "y": y_um,
                "width": cell_um,
                "height": cell_um,
                "fill": "none",
                "stroke": "#000000",
                "strokeWidth": 1.0
            })

            # label like A0, B0, ..., AA0
            label_items.append({
                "type": "text",                 # <-- required for renderer
                "units": "microns",
                "x": x_um + font_um,
                "y": y_um + font_um * 1.5,
                "text": f"{index_to_col(ix)}{iy}",
                "fill": "#000000",
                "fontSizeUm": font_um
            })

    return grid_items, label_items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Create a metric overlay grid (+labels) JSON for the SEM viewer."
    )
    dim = p.add_mutually_exclusive_group(required=True)
    dim.add_argument("--data-folder", type=str,
                     help="Folder containing summary_table.csv (with the standard header).")
    dim.add_argument("--width-mm", type=float,
                     help="Total width of stitched area in mm (use with --height-mm).")
    p.add_argument("--height-mm", type=float,
                   help="Total height of stitched area in mm (use with --width-mm).")

    p.add_argument("--cell-mm", type=float, default=1.0,
                   help="Grid square size in mm (default: 1.0).")
    p.add_argument("--font-mm", type=float, default=0.15,
                   help="Label height in mm (default: 0.15).")

    p.add_argument("--out-dir", type=str, default=".",
                   help="Output directory (default: current dir).")
    p.add_argument("--single-file", type=str, default=None,
                   help="Optional explicit path for a single overlay JSON. "
                        "If not set, defaults to <out-dir>/metric_overlay.json.")
    p.add_argument("--separate", action="store_true",
                   help="Write two files: metric_grid.json and metric_labels.json. "
                        "By default a single file is written.")

    args = p.parse_args()

    # Resolve dimensions
    if args.data_folder:
        w_mm, h_mm = dims_from_summary(Path(args.data_folder))
    else:
        if args.width_mm is None or args.height_mm is None:
            raise SystemExit("--width-mm and --height-mm must both be provided when not using --data-folder.")
        w_mm, h_mm = float(args.width_mm), float(args.height_mm)

    grid_items, label_items = build_metric_grid(
        width_mm=w_mm,
        height_mm=h_mm,
        cell_mm=args.cell_mm,
        font_mm=args.font_mm,
    )

    # Determine the base folder to anchor overlays
    if args.data_folder:
        # Always store overlays inside the data folder
        base_folder = Path(args.data_folder)
    else:
        # Fall back to user-provided out-dir
        base_folder = Path(args.out_dir)

    # overlays subfolder
    out_dir = base_folder / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)


    # Attach gating metadata so the viewer shows these only after origin is set
    meta = {
        "units": "microns",
        "relativeTo": "origin",
        "requiresOrigin": True
    }

    if args.separate:
        grid_path = out_dir / "metric_grid.json"
        labels_path = out_dir / "metric_labels.json"
        grid_path.write_text(json.dumps({**meta, "items": grid_items}, indent=2))
        labels_path.write_text(json.dumps({**meta, "items": label_items}, indent=2))
        print(f"wrote {grid_path}")
        print(f"wrote {labels_path}")
    else:
        # Default: single overlay file combining rects + text in one flat "items" list
        out_path = Path(args.single_file) if args.single_file else (out_dir / "metric_overlay.json")
        merged = [*grid_items, *label_items]
        out_path.write_text(json.dumps({**meta, "items": merged}, indent=2))
        print(f"wrote {out_path} (single overlay)")

if __name__ == "__main__":
    main()

