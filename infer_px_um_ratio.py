#!/usr/bin/env python3
"""
infer_px_um_ratio.py

Infer pixel size in µm/px (px_x_um, px_y_um) for an NDTiff dataset by
combining summary_table.csv (TileWidth_um/TileHeight_um) with actual
tile pixel dimensions read from the NDTiff frames.

Usage:
  python3 infer_px_um_ratio.py /path/to/ndtiff \
      [--z 0] [--max-rows 32] [--format text|json|shell] [--rtol 0.2] [--atol 0.02]

Notes:
- Requires: pandas, numpy, ndstorage
- We only read frames from a single Z (default 0); for most datasets that’s enough.
- If PosIndex in CSV looks 1-based, we normalize to 0-based automatically.
- If columns are missing or inference is unreliable, exits with non-zero status.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ndstorage import Dataset


def infer_px_um(ndtiff_folder: Path,
                z_for_probe: int = 0,
                max_rows: int = 32,
                rtol: float = 0.20,
                atol: float = 0.02) -> tuple[float, float]:
    """
    Return (px_x_um, px_y_um) or raise RuntimeError on failure.

    rtol: allowable relative mismatch between inferred X and Y (e.g. 0.20 = 20%)
    atol: allowable absolute mismatch (µm/px) between X and Y
    """
    csv_path = ndtiff_folder / "summary_table.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Must have size columns
    for c in ("TileWidth_um", "TileHeight_um"):
        if c not in df.columns or not df[c].notna().any():
            raise RuntimeError(f"Column '{c}' missing or empty in {csv_path}")

    # Open dataset & array (lazy)
    ds = Dataset(str(ndtiff_folder))
    images = ds.as_array()

    # Normalize PosIndex if present (convert 1-based -> 0-based)
    if "PosIndex" in df.columns:
        try:
            npos = images.shape[0]
            if df["PosIndex"].min() == 1 and df["PosIndex"].max() == npos:
                df = df.copy()
                df["PosIndex"] = df["PosIndex"] - 1
        except Exception:
            pass

    # Choose rows to probe (prefer the requested Z if column exists)
    if "Z_layer" in df.columns:
        dfz = df[df["Z_layer"] == z_for_probe]
        if dfz.empty:
            # fall back to any rows
            dfz = df
    else:
        dfz = df

    # Sample a reasonable number of rows
    if len(dfz) > max_rows:
        sample = dfz.sample(max_rows, random_state=0)
    else:
        sample = dfz

    vals_x = []
    vals_y = []

    for r in sample.itertuples(index=False):
        # Map to a position index
        if "PosIndex" in df.columns:
            try:
                pos = int(getattr(r, "PosIndex"))
            except Exception:
                continue
        elif "Index" in df.columns:
            # If Slices unknown, assume 1 (we only need the frame size)
            try:
                pos = int(getattr(r, "Index"))
            except Exception:
                continue
        else:
            # No usable index in CSV
            continue

        # Read one frame to get H_px, W_px
        try:
            a = images[pos, 0, 0, z_for_probe]
            try:
                a = a.compute()
            except Exception:
                pass
            a = np.asarray(a)
            # Reduce singleton-leading dims if present
            while a.ndim > 3 and a.shape[0] == 1:
                a = a[0]

            if a.ndim == 2:
                H_px, W_px = a.shape
            else:
                H_px, W_px = a.shape[0], a.shape[1]

            tw_um = float(getattr(r, "TileWidth_um"))
            th_um = float(getattr(r, "TileHeight_um"))
            if W_px > 0 and H_px > 0 and tw_um > 0 and th_um > 0:
                vals_x.append(tw_um / W_px)  # µm/px along X
                vals_y.append(th_um / H_px)  # µm/px along Y
        except Exception:
            continue

    if not vals_x or not vals_y:
        raise RuntimeError("Could not collect any valid (um, px) pairs to infer pixel size.")

    # Robust estimate via medians
    px_x_um = float(np.median(vals_x))
    px_y_um = float(np.median(vals_y))

    # Basic sanity checks
    if not (0.001 <= px_x_um <= 100.0 and 0.001 <= px_y_um <= 100.0):
        raise RuntimeError(f"Inferred values out of range: px_x_um={px_x_um}, px_y_um={px_y_um}")

    # Cross-consistency between X and Y
    if abs(px_x_um - px_y_um) > max(atol, rtol * max(px_x_um, px_y_um)):
        raise RuntimeError(
            f"X/Y mismatch too large: px_x_um={px_x_um:.6f}, px_y_um={px_y_um:.6f} "
            f"(rtol={rtol}, atol={atol})"
        )

    return px_x_um, px_y_um


def main():
    ap = argparse.ArgumentParser(description="Infer µm/px for an NDTiff dataset from CSV + frames.")
    ap.add_argument("ndtiff_folder", help="Path to the NDTiff dataset directory")
    ap.add_argument("--z", type=int, default=0, help="Z layer to probe (default: 0)")
    ap.add_argument("--max-rows", type=int, default=32, help="Max rows to sample for inference")
    ap.add_argument("--rtol", type=float, default=0.20, help="Relative tolerance between X and Y (default: 0.20)")
    ap.add_argument("--atol", type=float, default=0.02, help="Absolute tolerance between X and Y in µm/px (default: 0.02)")
    ap.add_argument("--format", choices=["text", "json", "shell"], default="text",
                    help="Output format: human text, JSON, or shell exports")
    args = ap.parse_args()

    ndpath = Path(args.ndtiff_folder)
    if not ndpath.is_dir():
        print(f"Folder not found: {ndpath}")
        raise SystemExit(2)

    try:
        px_x_um, px_y_um = infer_px_um(
            ndpath,
            z_for_probe=args.z,
            max_rows=args.max_rows,
            rtol=args.rtol,
            atol=args.atol,
        )
    except RuntimeError as e:
        print(f"[error] {e}")
        raise SystemExit(1)

    if args.format == "text":
        print(f"px_x_um={px_x_um:.6f}")
        print(f"px_y_um={px_y_um:.6f}")
    elif args.format == "json":
        print(json.dumps({"px_x_um": px_x_um, "px_y_um": px_y_um}, indent=2))
    else:  # shell
        # safe for: bash/zsh/fish (fish ignores 'export' but it's harmless)
        print(f"export PX_X_UM={px_x_um:.9f}")
        print(f"export PX_Y_UM={px_y_um:.9f}")


if __name__ == "__main__":
    main()

