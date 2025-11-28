#!/usr/bin/env python3
import csv, json, argparse, os

def main():
    ap = argparse.ArgumentParser(
        description="Create stitching-grid.json from summary_table.csv"
    )
    ap.add_argument("input_csv", help="summary_table.csv CSV file with X_rel_um, Y_rel_um, TileWidth_um, TileHeight_um")
    ap.add_argument("output_json", nargs="?", help="Output path (default: stitching-grid.json next to input)")
    args = ap.parse_args()

    out_path = args.output_json or os.path.join(
        os.path.dirname(os.path.abspath(args.input_csv)),
        "stitching-grid.json"
    )

    items = []
    with open(args.input_csv, newline="", encoding="utf-8") as f:
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

