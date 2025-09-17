#!/usr/bin/env python3
"""
Step 1 — EDS aggregation from NPZ + metadata

Inputs
------
- --eds-folder: directory containing files like "<base>_eds.npz"
  Each NPZ must contain key "eds_data" (y, x, channels). If missing, the
  first array in the NPZ is used as a fallback.
- --metadata-folder: directory with "<base>_metadata.txt" key=value lines
  (expects Stage Position X/Y, X/Y Cells, X/Y Step).

What it does
------------
For each EDS map:
  1) Load EDS cube (H x W x C) and the matching metadata.
  2) Split into an N x N grid (default 3 x 3).
  3) Sum spectra within each tile → 1D spectrum per tile.
  4) Save:
     - Per-tile spectra: processeddata/spectradata/<base>_seg-r{r}_c{c}.npz
     - All tiles together: processeddata/spectradata/<base>_all-segments.npz
     - A grid PNG of the spectra: processeddata/spectraplots/<base>_grid.png
     - A per-file tile table with normalized window intensity:
       processeddata/tiles/<base>_channel_<start>_to_<end>.txt
       (columns: base, stage_x_um, stage_y_um, row, col, frac_window_of_total)

Notes
-----
- Stage positions are written in micrometers (assuming metadata is in mm → ×1000).
- If your converter saved EDS as uint8, this still works; for quant work,
  consider re-exporting at higher bit depth later.
"""

from __future__ import annotations
import argparse
import sys
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# -------------------------- Metadata helpers --------------------------

def _parse_metadata_txt(path: Path) -> dict[str, str]:
    """
    Parse a simple metadata txt with 'key = value' lines.
    Keeps raw strings; use helpers below to pull numeric values.
    """
    meta = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def _get_first_numeric(meta: dict[str, str], key_suffix: str, default: float | None = None) -> float | None:
    """
    Find the first key whose name ends with `key_suffix` (case-insensitive)
    and return it as float if possible; otherwise `default`.
    """
    ksuf = key_suffix.lower()
    for k, v in meta.items():
        if k.lower().endswith(ksuf):
            # extract the first numeric token in v
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    pass
    return default


def read_stage_positions_um(meta: dict[str, str]) -> tuple[float | None, float | None]:
    """
    Returns stage X, Y in micrometers, assuming metadata is stored in mm.
    If your metadata is already in µm, change the scale factor to 1.0.
    """
    x_mm = _get_first_numeric(meta, "Stage Position/X")
    y_mm = _get_first_numeric(meta, "Stage Position/Y")
    if x_mm is None or y_mm is None:
        return None, None
    return x_mm * 1000.0, y_mm * 1000.0


# --------------------------- Core processing ---------------------------

def load_eds_cube(npz_path: Path) -> np.ndarray:
    """
    Load an EDS cube (H, W, C) from NPZ.
    Prefers 'eds_data' but falls back to the first array in the file.
    """
    z = np.load(npz_path)
    if "eds_data" in z:
        arr = z["eds_data"]
    else:
        # fallback to the first array key (np.savez unnamed => arr_0)
        first_key = z.files[0]
        arr = z[first_key]
    if arr.ndim != 3:
        raise ValueError(f"{npz_path.name}: expected 3D array (H,W,C), got shape {arr.shape}")
    return arr


def split_edges(n: int, length: int) -> np.ndarray:
    """Integer edges for splitting an axis of `length` into `n` nearly equal parts."""
    return np.linspace(0, length, n + 1, dtype=int)


def aggregate_tiles(eds: np.ndarray, grid: int) -> np.ndarray:
    """
    Sum spectra per tile over an N x N grid.

    Returns:
      spectra: (grid, grid, C) array
    """
    H, W, C = eds.shape
    y_edges = split_edges(grid, H)
    x_edges = split_edges(grid, W)

    spectra = np.zeros((grid, grid, C), dtype=np.float64)
    for r in range(grid):
        ys, ye = y_edges[r], y_edges[r + 1]
        for c in range(grid):
            xs, xe = x_edges[c], x_edges[c + 1]
            tile = eds[ys:ye, xs:xe, :]
            # sum over pixels to 1D spectrum
            spectra[r, c, :] = tile.sum(axis=(0, 1), dtype=np.float64)
    return spectra


def save_tile_table(base: str,
                    out_tiles_dir: Path,
                    spectra_rc: np.ndarray,
                    window_start: int,
                    window_end: int,
                    stage_x_um: float | None,
                    stage_y_um: float | None) -> Path:
    """
    Write a per-file tile table with normalized window intensities.
    """
    out_tiles_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_tiles_dir / f"{base}_channel_{window_start}_to_{window_end}.txt"

    # header
    header = (
        "# base, stage_x_um, stage_y_um, row, col, frac_window_of_total\n"
        f"# window_start={window_start}, window_end={window_end}\n"
    )
    out_txt.write_text(header, encoding="utf-8")

    G, _, C = spectra_rc.shape
    ws = max(0, min(C, window_start))
    we = max(ws, min(C, window_end))  # half-open [ws, we)
    for r in range(G):
        for c in range(G):
            spec = spectra_rc[r, c, :]
            total = spec.sum()
            win = spec[ws:we].sum() if we > ws else 0.0
            frac = float(win / total) if total > 0 else 0.0
            line = f"{base}, {stage_x_um if stage_x_um is not None else ''}, {stage_y_um if stage_y_um is not None else ''}, {r}, {c}, {frac:.6g}\n"
            with out_txt.open("a", encoding="utf-8") as f:
                f.write(line)
    return out_txt


def plot_grid_of_spectra(base: str,
                         spectra_rc: np.ndarray,
                         out_plot_dir: Path) -> Path:
    """
    Make an N x N subplot grid of the per-tile spectra.
    """
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    G, _, C = spectra_rc.shape

    fig, axs = plt.subplots(G, G, figsize=(4 * G, 3 * G), squeeze=False)
    x = np.arange(C)
    for r in range(G):
        for c in range(G):
            ax = axs[r][c]
            ax.plot(x, spectra_rc[r, c, :])
            ax.set_title(f"Tile ({r}, {c})")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Counts")
            ax.margins(x=0)
    fig.suptitle(base, y=0.995)
    fig.tight_layout()

    out_png = out_plot_dir / f"{base}_grid.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_per_tile_npz(base: str,
                      spectra_rc: np.ndarray,
                      out_spec_dir: Path) -> None:
    """
    Save all tiles together, and also one NPZ per tile for convenience.
    """
    out_spec_dir.mkdir(parents=True, exist_ok=True)
    # Save the full stack
    np.savez_compressed(
        out_spec_dir / f"{base}_all-segments.npz",
        spectra=spectra_rc
    )
    # Save per-tile
    G, _, _ = spectra_rc.shape
    for r in range(G):
        for c in range(G):
            np.savez_compressed(
                out_spec_dir / f"{base}_seg-r{r}_c{c}.npz",
                spectrum=spectra_rc[r, c, :]
            )


# ------------------------------ CLI -----------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate EDS spectra per tile from NPZ + metadata.")
    p.add_argument("--eds-folder", required=True, type=Path,
                   help="Folder with *_eds.npz files.")
    p.add_argument("--metadata-folder", required=True, type=Path,
                   help="Folder with *_metadata.txt files.")
    p.add_argument("--grid", type=int, default=3,
                   help="Grid size N for N×N tiling (default: 3).")
    p.add_argument("--window-start", type=int, default=500,
                   help="Start channel (inclusive) for normalized window integral (default: 500).")
    p.add_argument("--window-end", type=int, default=1000,
                   help="End channel (exclusive) for normalized window integral (default: 1000).")
    p.add_argument("--outdir", type=Path, default=Path("processeddata"),
                   help="Output base folder (default: processeddata).")
    p.add_argument("--glob", type=str, default="*_eds.npz",
                   help="Glob pattern to match EDS files (default: '*_eds.npz').")
    args = p.parse_args(argv)

    eds_dir: Path = args.eds_folder
    meta_dir: Path = args.metadata_folder
    out_base: Path = args.outdir
    out_spec = out_base / "spectradata"
    out_plot = out_base / "spectraplots"
    out_tiles = out_base / "tiles"

    files = sorted(eds_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No EDS files found in {eds_dir} with pattern {args.glob}", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(files)} EDS file(s). Grid={args.grid}, window=[{args.window_start},{args.window_end})")
    out_spec.mkdir(parents=True, exist_ok=True)
    out_plot.mkdir(parents=True, exist_ok=True)
    out_tiles.mkdir(parents=True, exist_ok=True)

    for i, eds_path in enumerate(files, 1):
        base_full = eds_path.stem  # e.g., "..._eds"
        base = base_full[:-4] if base_full.endswith("_eds") else base_full

        # Find matching metadata
        meta_path = meta_dir / f"{base}_metadata.txt"
        if not meta_path.exists():
            # fallback: try exact stem match with '_metadata.txt' ignoring '_eds'
            candidates = list(meta_dir.glob(f"{base}*_metadata.txt"))
            if candidates:
                meta_path = candidates[0]

        print(f"[{i}/{len(files)}] {eds_path.name}")
        try:
            eds = load_eds_cube(eds_path)
        except Exception as e:
            print(f"  [ERROR] Failed to load EDS cube: {e}", file=sys.stderr)
            continue

        meta = _parse_metadata_txt(meta_path)
        if not meta:
            print(f"  [WARN] Metadata not found or empty: {meta_path.name}")
        stage_x_um, stage_y_um = read_stage_positions_um(meta)

        # Sanity check: if X/Y Cells recorded, compare to array shape
        x_cells = _get_first_numeric(meta, "X Cells")
        y_cells = _get_first_numeric(meta, "Y Cells")
        H, W, C = eds.shape
        if x_cells and int(round(x_cells)) != W:
            print(f"  [NOTE] Metadata X Cells={x_cells} vs data W={W}")
        if y_cells and int(round(y_cells)) != H:
            print(f"  [NOTE] Metadata Y Cells={y_cells} vs data H={H}")

        # Aggregate per tile
        spectra_rc = aggregate_tiles(eds, args.grid)

        # Save outputs
        save_per_tile_npz(base, spectra_rc, out_spec)
        plot_grid_of_spectra(base, spectra_rc, out_plot)
        txt_path = save_tile_table(base, out_tiles, spectra_rc,
                                   args.window_start, args.window_end,
                                   stage_x_um, stage_y_um)

        print(f"  -> spectra:   {out_spec / (base + '_all-segments.npz')}")
        print(f"  -> plot:      {out_plot / (base + '_grid.png')}")
        print(f"  -> tile tbl:  {txt_path}")

    print("[DONE] All files processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
