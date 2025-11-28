#!/usr/bin/env python3
import argparse, pathlib, csv, json, numpy as np

def load_cube(p: pathlib.Path) -> np.ndarray:
    with np.load(p, allow_pickle=False) as z:
        a = np.asarray(z[next(iter(z.files))])
    if a.ndim != 3:
        raise ValueError(f"{p.name}: expected 3D counts cube, got {a.shape}")
    # ensure channels-last (H, W, C) if needed
    if a.shape[0] < min(a.shape[1], a.shape[2]):  # likely (C,H,W)
        a = np.moveaxis(a, 0, -1)
    return a

def window_sums_cropped(cube: np.ndarray, rows: int, cols: int,
                        right_off_pct: float, bottom_off_pct: float):
    """
    Return (rows, cols, channels) int64 array of window-summed spectra,
    computed over the top-left (1-right_off_pct) x (1-bottom_off_pct) area.
    Also returns the effective pixel extent (H_eff, W_eff).
    """
    H, W, C = cube.shape
    W_eff = int(round(W * (1.0 - right_off_pct / 100.0)))
    H_eff = int(round(H * (1.0 - bottom_off_pct / 100.0)))
    W_eff = max(1, min(W_eff, W))
    H_eff = max(1, min(H_eff, H))

    ys = np.linspace(0, H_eff, rows + 1, dtype=int)
    xs = np.linspace(0, W_eff, cols + 1, dtype=int)
    out = np.empty((rows, cols, C), dtype=np.int64)
    for r in range(rows):
        y0, y1 = ys[r], ys[r + 1]
        for c in range(cols):
            x0, x1 = xs[c], xs[c + 1]
            # sum within cropped window only
            out[r, c] = cube[y0:y1, x0:x1, :].sum((0, 1))
    return out, (H_eff, W_eff)

def main():
    ap = argparse.ArgumentParser(description="Per-tile JSON with per-window aggregated spectra + stage/meta, with right/bottom crop.")
    ap.add_argument("h5data_dir", help="Folder containing summary_table.csv and *_eds.npz")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--right-offset-percent", type=float, default=0.0,
                    help="Crop this percent from the RIGHT of each tile before grid (e.g. 10)")
    ap.add_argument("--bottom-offset-percent", type=float, default=0.0,
                    help="Crop this percent from the BOTTOM of each tile before grid (e.g. 10)")
    args = ap.parse_args()

    if not (0.0 <= args.right_offset_percent < 100.0 and 0.0 <= args.bottom_offset_percent < 100.0):
        raise SystemExit("Offset percents must be in [0, 100).")

    root = pathlib.Path(args.h5data_dir)
    summary_csv = root / "summary_table.csv"
    if not summary_csv.exists():
        raise SystemExit(f"Missing {summary_csv}")

    # Load summary table (exact columns)
    summary = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        need = ["basename", "X_rel_um", "Y_rel_um", "TileWidth_um", "TileHeight_um"]
        if rdr.fieldnames is None or any(n not in rdr.fieldnames for n in need):
            raise SystemExit(f"CSV must have columns: {need}")
        for row in rdr:
            base = row["basename"]
            summary[base] = {
                "X_rel_um": float(row["X_rel_um"]),
                "Y_rel_um": float(row["Y_rel_um"]),
                "TileWidth_um": float(row["TileWidth_um"]),
                "TileHeight_um": float(row["TileHeight_um"]),
            }

    out_dir = root / f"aggregated-spectra"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(root.rglob("*_eds.npz"))
    if not npz_files:
        raise SystemExit(f"No '*_eds.npz' under {root}")

    total = len(npz_files)
    processed = skipped = 0
    for i, npz in enumerate(npz_files, 1):
        print(f"[{i}/{total}] {npz.name} …", flush=True)
        basename = npz.name[:-8]  # strip "_eds.npz"

        meta = summary.get(basename)
        if meta is None:
            print(f"  -> skip: '{basename}' not found in summary_table.csv")
            skipped += 1
            continue

        try:
            cube = load_cube(npz)
        except Exception as e:
            print(f"  -> skip: {e}")
            skipped += 1
            continue

        # Aggregate spectra over a cropped top-left area
        specs, (H_eff, W_eff) = window_sums_cropped(
            cube, args.rows, args.cols,
            args.right_offset_percent, args.bottom_offset_percent
        )
        R, C, K = specs.shape

        # Compute quadrant geometry in µm relative to tile origin
        # Scale tile um dimensions by the same crop percents:
        tile_width_eff_um  = meta["TileWidth_um"]  * (1.0 - args.right_offset_percent  / 100.0)
        tile_height_eff_um = meta["TileHeight_um"] * (1.0 - args.bottom_offset_percent / 100.0)
        quad_w_um = tile_width_eff_um  / C
        quad_h_um = tile_height_eff_um / R

        entries = []
        for r in range(R):
            for c in range(C):
                entries.append({
                    "basename": basename,
                    "X_rel_um": meta["X_rel_um"],      # tile origin (absolute)
                    "Y_rel_um": meta["Y_rel_um"],
                    "TileWidth_um": meta["TileWidth_um"],
                    "TileHeight_um": meta["TileHeight_um"],
                    "rows": R, "cols": C,
                    "right_offset_percent": args.right_offset_percent,
                    "bottom_offset_percent": args.bottom_offset_percent,
                    "rownum": r, "colnum": c,

                    # Quadrant geometry (µm) RELATIVE TO TILE ORIGIN (top-left)
                    "quad_x_um": c * quad_w_um,
                    "quad_y_um": r * quad_h_um,
                    "quad_width_um": quad_w_um,
                    "quad_height_um": quad_h_um,

                    # Aggregated spectrum (list of channel counts)
                    "aggregatedspectrum": specs[r, c].tolist()
                })

        out_json = out_dir / f"{basename}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(entries, f, separators=(",", ":"))
        processed += 1
        print(f"  -> wrote {out_json.name}", flush=True)

    print(f"Done. processed={processed}, skipped={skipped}, out={out_dir}")

if __name__ == "__main__":
    main()

