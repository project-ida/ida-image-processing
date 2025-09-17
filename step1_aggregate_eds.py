#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, re, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------- Metadata helpers --------------------------

def _parse_metadata_txt(path: Path) -> dict[str, str]:
    """
    Accepts both 'key = value' and 'key: value' lines.
    """
    meta = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line or ":" in line:
            try:
                k, v = re.split(r"\s*[=:]\s*", line, maxsplit=1)
                meta[k.strip()] = v.strip()
            except Exception:
                continue
    return meta

def _get_first_numeric(meta: dict[str, str], key_suffix: str, default: float | None = None) -> float | None:
    ksuf = key_suffix.lower()
    for k, v in meta.items():
        if k.lower().endswith(ksuf):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    pass
    return default

def read_stage_positions_um(meta: dict[str, str]) -> tuple[float | None, float | None]:
    x_mm = _get_first_numeric(meta, "Stage Position/X")
    y_mm = _get_first_numeric(meta, "Stage Position/Y")
    if x_mm is None or y_mm is None:
        return None, None
    return x_mm * 1000.0, y_mm * 1000.0

# --------------------------- Core processing ---------------------------

def _open_npz_safely(p: Path):
    """
    Open an NPZ, returning (npz, error_str). error_str is None if ok.
    """
    try:
        z = np.load(p)
        # Touch a tiny read to trigger zip integrity early.
        _ = z.files
        return z, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def load_eds_cube(npz_path: Path) -> np.ndarray:
    """
    Load an EDS cube (H, W, C). Prefer key 'eds_data', else first array.
    Raises ValueError if not possible.
    """
    z, err = _open_npz_safely(npz_path)
    if err:
        raise ValueError(f"Cannot open {npz_path.name}: {err}")
    try:
        if "eds_data" in z:
            arr = z["eds_data"]
        else:
            key = z.files[0]
            arr = z[key]
    finally:
        z.close()
    if arr.ndim != 3:
        raise ValueError(f"{npz_path.name}: expected 3D array (H,W,C), got {arr.shape}")
    return arr

def split_edges(n: int, length: int) -> np.ndarray:
    return np.linspace(0, length, n + 1, dtype=int)

def aggregate_tiles(eds: np.ndarray, grid: int) -> np.ndarray:
    H, W, C = eds.shape
    y_edges = split_edges(grid, H)
    x_edges = split_edges(grid, W)
    spectra = np.zeros((grid, grid, C), dtype=np.float64)
    for r in range(grid):
        ys, ye = y_edges[r], y_edges[r + 1]
        for c in range(grid):
            xs, xe = x_edges[c], x_edges[c + 1]
            tile = eds[ys:ye, xs:xe, :]
            spectra[r, c, :] = tile.sum(axis=(0, 1), dtype=np.float64)
    return spectra

def save_tile_table(base: str, out_tiles_dir: Path, spectra_rc: np.ndarray,
                    window_start: int, window_end: int,
                    stage_x_um: float | None, stage_y_um: float | None,
                    meta_path: Path | None) -> Path:
    out_tiles_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_tiles_dir / f"{base}_channel_{window_start}_to_{window_end}.txt"
    header = (
        "# base, stage_x_um, stage_y_um, row, col, frac_window_of_total\n"
        f"# window_start={window_start}, window_end={window_end}\n"
        f"# metadata={meta_path if meta_path else 'N/A'}\n"
    )
    out_txt.write_text(header, encoding="utf-8")
    G, _, C = spectra_rc.shape
    ws = max(0, min(C, window_start)); we = max(ws, min(C, window_end))
    for r in range(G):
        for c in range(G):
            spec = spectra_rc[r, c, :]
            total = spec.sum()
            win = spec[ws:we].sum() if we > ws else 0.0
            frac = float(win / total) if total > 0 else 0.0
            with out_txt.open("a", encoding="utf-8") as f:
                f.write(f"{base}, {stage_x_um if stage_x_um is not None else ''}, "
                        f"{stage_y_um if stage_y_um is not None else ''}, "
                        f"{r}, {c}, {frac:.6g}\n")
    return out_txt

def plot_grid_of_spectra(base: str, spectra_rc: np.ndarray, out_plot_dir: Path) -> Path:
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    G, _, C = spectra_rc.shape
    fig, axs = plt.subplots(G, G, figsize=(4 * G, 3 * G), squeeze=False)
    x = np.arange(C)
    for r in range(G):
        for c in range(G):
            ax = axs[r][c]
            ax.plot(x, spectra_rc[r, c, :])
            ax.set_title(f"Tile ({r}, {c})")
            ax.set_xlabel("Channel"); ax.set_ylabel("Counts"); ax.margins(x=0)
    fig.suptitle(base, y=0.995); fig.tight_layout()
    out_png = out_plot_dir / f"{base}_grid.png"
    fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

def save_per_tile_npz(base: str, spectra_rc: np.ndarray, out_spec_dir: Path) -> None:
    out_spec_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_spec_dir / f"{base}_all-segments.npz", spectra=spectra_rc)
    G, _, _ = spectra_rc.shape
    for r in range(G):
        for c in range(G):
            np.savez_compressed(out_spec_dir / f"{base}_seg-r{r}_c{c}.npz",
                                spectrum=spectra_rc[r, c, :])

# ------------------------------ CLI -----------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate EDS spectra per tile from NPZ + metadata.")
    p.add_argument("--eds-folder", required=True, type=Path)
    p.add_argument("--metadata-folder", required=True, type=Path)
    p.add_argument("--grid", type=int, default=3)
    p.add_argument("--window-start", type=int, default=500)
    p.add_argument("--window-end", type=int, default=1000)
    p.add_argument("--outdir", type=Path, default=Path("processeddata"))
    p.add_argument("--glob", type=str, default="*_eds.npz")
    p.add_argument("--keep-going", action="store_true",
                   help="Skip corrupted EDS files instead of aborting.")
    args = p.parse_args(argv)

    eds_dir, meta_dir = args.eds_folder, args.metadata_folder
    out_spec, out_plot, out_tiles = (args.outdir / "spectradata",
                                     args.outdir / "spectraplots",
                                     args.outdir / "tiles")
    files = sorted(eds_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No EDS files found in {eds_dir} with pattern {args.glob}", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(files)} EDS file(s). Grid={args.grid}, window=[{args.window_start},{args.window_end})")
    out_spec.mkdir(parents=True, exist_ok=True)
    out_plot.mkdir(parents=True, exist_ok=True)
    out_tiles.mkdir(parents=True, exist_ok=True)

    for i, eds_path in enumerate(files, 1):
        base_full = eds_path.stem
        base = base_full[:-4] if base_full.endswith("_eds") else base_full

        # locate metadata
        meta_path = meta_dir / f"{base}_metadata.txt"
        origin = "exact"
        if not meta_path.exists():
            mp2 = eds_path.with_name(f"{base}_metadata.txt")
            if mp2.exists():
                meta_path, origin = mp2, "same-dir"
            else:
                hits = list(meta_dir.rglob(f"*{base}*_metadata.txt"))
                if hits:
                    meta_path, origin = hits[0], "rglob"
                else:
                    meta_path, origin = None, "missing"

        print(f"[{i}/{len(files)}] {eds_path.name}")
        try:
            eds = load_eds_cube(eds_path)
        except Exception as e:
            msg = f"  [ERROR] EDS load failed for {eds_path.name}: {e}"
            if args.keep-going:
                print(msg, file=sys.stderr); continue
            else:
                raise

        meta = _parse_metadata_txt(meta_path) if meta_path else {}
        if not meta:
            print(f"  [WARN] Metadata {'missing' if meta_path is None else 'empty/unparsed'} ({origin}): "
                  f"{(meta_path.name if meta_path else 'N/A')}")
        stage_x_um, stage_y_um = read_stage_positions_um(meta)

        x_cells = _get_first_numeric(meta, "X Cells"); y_cells = _get_first_numeric(meta, "Y Cells")
        H, W, C = eds.shape
        if x_cells and int(round(x_cells)) != W:
            print(f"  [NOTE] Metadata X Cells={x_cells} vs data W={W}")
        if y_cells and int(round(y_cells)) != H:
            print(f"  [NOTE] Metadata Y Cells={y_cells} vs data H={H}")

        spectra_rc = aggregate_tiles(eds, args.grid)
        save_per_tile_npz(base, spectra_rc, out_spec)
        plot_grid_of_spectra(base, spectra_rc, out_plot)
        txt_path = save_tile_table(base, out_tiles, spectra_rc,
                                   args.window_start, args.window_end,
                                   stage_x_um, stage_y_um, meta_path)
        print(f"  -> spectra:   {out_spec / (base + '_all-segments.npz')}")
        print(f"  -> plot:      {out_plot / (base + '_grid.png')}")
        print(f"  -> tile tbl:  {txt_path}")

    print("[DONE] All files processed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
