#!/usr/bin/env python3
import csv, json, argparse, os

def main():
    ap = argparse.ArgumentParser(
        description="Create stitching-grid.json from summary_table.csv"
    )
    ap.add_argument("input_folder", help="Folder containing summary_table.csv")
    ap.add_argument("output_json", nargs="?", help="Output filename (stored in overlays/ inside the folder)")
    args = ap.parse_args()

    # input_folder contains a CSV whose name is always summary_table.csv    
    folder = os.path.abspath(args.input_folder)
    csv_path = os.path.join(folder, "summary_table.csv")

    if not os.path.exists(csv_path):
        raise SystemExit(f"summary_table.csv not found in {folder}")

    # output is always under overlays/ inside the folder
    overlays_dir = os.path.join(folder, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    if args.output_json:
        out_path = os.path.join(overlays_dir, args.output_json)
    else:
        out_path = os.path.join(overlays_dir, "stitching-grid.json")

    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append({
                "type": "rect",
                "units": "microns",             # absolute µm; ignores sample origin in the UI
                "x": float(row["X_rel_um"]),
                "y": float(row["Y_rel_um"]),
                "width": float(row["TileWidth_um"]),
                "height": float(row["TileHeight_um"]),
                "fill": "none",                 # no fill
                "stroke": "#e53935",            # thin red outline
                "strokeWidth": 1,               # interpreted as px; app normalizes to width
                "strokeDasharray": "3,3"        # dashed line
            })

    # optional: sort top-left → bottom-right (not required)
    items.sort(key=lambda r: (r["y"], r["x"]))

    # file-level wrapper lets the viewer default units for all items
    payload = {"units": "microns", "items": items}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(items)} rectangles → {out_path}")

if __name__ == "__main__":
    main()
