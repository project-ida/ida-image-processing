#!/usr/bin/env python3
"""
Step 2 — SEM image → tiles + Fiji TileConfiguration.txt
Now with:
- --norm global      : one contrast range for ALL tiles (auto-chosen by percentiles)
- Uses existing knobs: --auto-clip-percent (tails to ignore), --gamma
- Keeps: tiff16/png8 output, invert-x/y, auto-um-per-px, tile audit (+ neighbors),
         corner sanity print, TileConfiguration.txt written into tiles folder.

Tip: For uniform + punchy images start with:
  --norm global --auto-clip-percent 1.0 --gamma 0.9
"""

from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

# -------------------------- Metadata helpers --------------------------

def parse_metadata_txt(path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s or ":" in s:
            try:
                k, v = re.split(r"\s*[=:]\s*", s, maxsplit=1)
                meta[k.strip()] = v.strip()
            except Exception:
                continue
    return meta

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def meta_num(meta: Dict[str, str], key_suffix: str, default: float | None = None) -> float | None:
    ksuf = key_suffix.lower()
    for k, v in meta.items():
        if k.lower().endswith(ksuf):
            m = _num_re.search(v)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    pass
    return default

# ---------------------------- Image I/O --------------------------------

def load_sem_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as z:
        key = "sem_data" if "sem_data" in z.files else (z.files[0] if z.files else None)
        if key is None:
            raise ValueError("empty NPZ")
        arr = z[key]
    if arr.ndim != 2:
        raise ValueError(f"{npz_path.name}: expected 2D array, got shape {arr.shape}")
    return arr

def compute_lo_hi(arr: np.ndarray, method: str,
                  vmin: float | None, vmax: float | None,
                  auto_clip_percent: float) -> Tuple[float, float]:
    a = arr.astype(np.float64, copy=False)
    if method == "auto":
        if auto_clip_percent and auto_clip_percent > 0:
            lo, hi = np.nanpercentile(a, [auto_clip_percent, 100.0 - auto_clip_percent])
        else:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    elif method == "fixed":
        if vmin is None or vmax is None or vmax <= vmin:
            raise ValueError("--norm fixed requires valid --vmin < --vmax")
        lo, hi = float(vmin), float(vmax)
    elif method == "absolute16":
        lo, hi = 0.0, 65535.0
    elif method == "absolute":
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype); lo, hi = float(info.min), float(info.max)
        else:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    else:
        raise ValueError(f"Unknown --norm '{method}'")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return lo, hi

def scale_to_uint8(arr: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    a = arr.astype(np.float64, copy=False)
    a = (a - lo) / max(hi - lo, 1e-12)
    a = np.clip(a, 0.0, 1.0)
    if gamma and gamma != 1.0:
        a = np.power(a, gamma)
    return (a * 255.0 + 0.5).astype(np.uint8)

def scale_to_uint16(arr: np.ndarray, method: str, lo: float, hi: float, gamma: float) -> np.ndarray:
    if method == "absolute16":
        return np.clip(arr, 0.0, 65535.0).astype(np.uint16)
    a = arr.astype(np.float64, copy=False)
    a = (a - lo) / max(hi - lo, 1e-12)
    a = np.clip(a, 0.0, 1.0)
    if gamma and gamma != 1.0:
        a = np.power(a, gamma)
    return (a * 65535.0 + 0.5).astype(np.uint16)

# ------------------------- Global range (new) --------------------------

def estimate_global_lo_hi(sem_paths: List[Path],
                          clip_percent: float,
                          sample_per_tile: int = 5000,
                          seed: int = 0) -> Tuple[float, float]:
    """
    Sample up to `sample_per_tile` pixels from each tile and compute
    robust global [lo,hi] using percentiles. clip_percent=1.0 → 1st/99th.
    """
    rng = np.random.default_rng(seed)
    samples: List[np.ndarray] = []
    for p in sem_paths:
        try:
            a = load_sem_npz(p).astype(np.float64, copy=False)
        except Exception:
            continue
        flat = a.ravel()
        if sample_per_tile and flat.size > sample_per_tile:
            idx = rng.choice(flat.size, sample_per_tile, replace=False)
            s = flat[idx]
        else:
            s = flat
        samples.append(s)
    if not samples:
        return 0.0, 1.0
    all_s = np.concatenate(samples)
    if clip_percent and clip_percent > 0:
        lo, hi = np.nanpercentile(all_s, [clip_percent, 100.0 - clip_percent])
    else:
        lo, hi = float(np.nanmin(all_s)), float(np.nanmax(all_s))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return float(lo), float(hi)

# ----------------------- TileConfiguration.txt -------------------------

def write_tileconfig(out_path: Path, entries: List[Tuple[str, float, float]]) -> None:
    lines = ["# Define the number of dimensions we are working on",
             "dim = 2", "", "# Define the image coordinates"]
    lines += [f"{name}; ; ({x:.3f}, {y:.3f})" for name, x, y in entries]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# --------------------------------- CLI ---------------------------------

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert SEM NPZ to tiles and build TileConfiguration.txt (+ audit)")
    p.add_argument("--sem-folder", required=True, type=Path)
    p.add_argument("--metadata-folder", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=Path("processeddata"))
    p.add_argument("--tiles-subdir", type=str, default="sem-images-png")
    p.add_argument("--tileconfig-name", type=str, default="TileConfiguration.txt")
    p.add_argument("--glob", type=str, default="*_sem.npz")
    # tiles + normalization
    p.add_argument("--out-format", choices=["png8", "tiff16"], default="png8")
    p.add_argument("--norm", choices=["auto", "absolute", "absolute16", "fixed", "global"], default="auto")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--auto-clip-percent", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--global-sample-per-tile", type=int, default=5000,
                   help="When --norm global, sample this many pixels per tile to estimate the range.")
    # mapping
    p.add_argument("--um-per-px", type=float, default=0.5490099)
    p.add_argument("--auto-um-per-px", action="store_true")
    p.add_argument("--invert-x", action="store_true")
    p.add_argument("--invert-y", action="store_true")
    args = p.parse_args(argv)

    sem_dir, meta_dir = args.sem_folder, args.metadata_folder
    outdir = args.outdir
    tiles_dir = outdir / args.tiles_subdir
    tiles_dir.mkdir(parents=True, exist_ok=True)

    sem_paths = sorted(sem_dir.glob(args.glob))
    if not sem_paths:
        print(f"[WARN] No SEM files found in {sem_dir} with pattern {args.glob}", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(sem_paths)} SEM file(s). out={args.out_format}, norm={args.norm}, "
          f"gamma={args.gamma}, clip={args.auto-clip-percent}")

    # -------- global normalization pre-pass --------
    global_lo_hi: Tuple[float, float] | None = None
    if args.norm == "global":
        glo, ghi = estimate_global_lo_hi(
            sem_paths, clip_percent=args.auto_clip_percent,
            sample_per_tile=args.global_sample_per_tile, seed=0
        )
        global_lo_hi = (glo, ghi)
        print(f"[INFO] global lo/hi (clip={args.auto_clip_percent}%): {glo:.6g}  {ghi:.6g}")

    # For audit/reporting
    all_tile_names: List[str] = []
    skipped_tiles: List[str] = []
    tile_sizes: List[Tuple[int, int]] = []
    step_um_x_candidates: List[float] = []
    step_um_y_candidates: List[float] = []
    records: List[Tuple[str, float, float]] = []  # (tile_name, x_um, y_um)

    # -------- tile export loop --------
    for i, sem_path in enumerate(sem_paths, 1):
        base_full = sem_path.stem
        base = base_full[:-4] if base_full.endswith("_sem") else base_full

        try:
            arr = load_sem_npz(sem_path)
        except Exception as e:
            print(f"[ERROR] {sem_path.name}: load failed → {e}", file=sys.stderr)
            continue

        H, W = arr.shape
        tile_sizes.append((W, H))

        # choose normalization range
        if args.norm == "global" and global_lo_hi is not None:
            lo, hi = global_lo_hi
        else:
            lo, hi = compute_lo_hi(arr, args.norm, args.vmin, args.vmax, args.auto_clip_percent)

        # export tile
        out_name = f"{base}_sem." + ("png" if args.out_format == "png8" else "tif")
        out_path = tiles_dir / out_name
        if args.out_format == "png8":
            img8 = scale_to_uint8(arr, lo, hi, args.gamma)
            Image.fromarray(img8, mode="L").save(out_path)
        else:
            img16 = scale_to_uint16(arr, args.norm, lo, hi, args.gamma)
            Image.fromarray(img16, mode="I;16").save(out_path)

        print(f"[{i}/{len(sem_paths)}] {out_name} ({W}×{H})  lo={lo:.3g} hi={hi:.3g} gamma={args.gamma}")
        all_tile_names.append(out_name)

        # metadata → stage + steps
        meta_path = meta_dir / f"{base}_metadata.txt"
        if not meta_path.exists():
            alt = sem_path.with_name(f"{base}_metadata.txt")
            if alt.exists(): meta_path = alt
        meta = parse_metadata_txt(meta_path) if meta_path.exists() else {}

        x_mm = meta_num(meta, "Stage Position/X")
        y_mm = meta_num(meta, "Stage Position/Y")
        if x_mm is None or y_mm is None:
            print(f"    [WARN] stage positions missing for {base} — excluded from TileConfiguration")
            skipped_tiles.append(out_name)
            continue

        x_step_mm = meta_num(meta, "X Step")
        y_step_mm = meta_num(meta, "Y Step")
        if x_step_mm: step_um_x_candidates.append(x_step_mm * 1000.0)
        if y_step_mm: step_um_y_candidates.append(y_step_mm * 1000.0)

        records.append((out_name, x_mm * 1000.0, y_mm * 1000.0))

    if not records:
        print("[WARN] No records with valid stage positions; TileConfiguration.txt will not be written.", file=sys.stderr)
        return 0

    # -------- µm/px selection --------
    um_per_px = args.um_per_px
    if args.auto_um_per_px:
        umppx_candidates = []
        if step_um_x_candidates and tile_sizes:
            w_px = int(np.median([w for w, _ in tile_sizes])); umppx_candidates.append(np.median(step_um_x_candidates) / max(w_px, 1))
        if step_um_y_candidates and tile_sizes:
            h_px = int(np.median([h for _, h in tile_sizes])); umppx_candidates.append(np.median(step_um_y_candidates) / max(h_px, 1))
        if umppx_candidates:
            um_per_px = float(np.median(umppx_candidates))
            print(f"[INFO] --auto-um-per-px → {um_per_px:.6f} µm/px (from X/Y Step & tile size)")
        else:
            print(f"[WARN] --auto-um-per-px requested but no X/Y Step found; using --um-per-px={um_per_px}", file=sys.stderr)
    else:
        print(f"[INFO] Using manual --um-per-px = {um_per_px}")

    # -------- coords → pixels & TileConfiguration --------
    names, xs_um, ys_um = zip(*records)
    xs = np.array(xs_um, dtype=float)
    ys = np.array(ys_um, dtype=float)
    if args.invert_x: xs = -xs
    if args.invert_y: ys = -ys
    xs -= xs.min(); ys -= ys.min()
    xs_px = xs / um_per_px; ys_px = ys / um_per_px

    order = np.lexsort((xs_px, ys_px))
    entries = [(names[i], float(xs_px[i]), float(ys_px[i])) for i in order]

    tile_path = tiles_dir / args.tileconfig_name
    write_tileconfig(tile_path, entries)
    print(f"[OK] Wrote {tile_path} with {len(entries)} entries")
    print("     Open from the same folder in Fiji; leave 'Invert X/Y' unchecked.")

    # ---------------------- Tile Audit (missing / duplicates) ----------------------
    audit_lines: List[str] = []
    audit_lines.append(f"Tiles written: {len(all_tile_names)}")
    audit_lines.append(f"Tiles in TileConfiguration (with stage positions): {len(names)}")
    if skipped_tiles:
        audit_lines.append(f"Skipped (no/invalid metadata): {len(skipped_tiles)}")
        preview = ', '.join(skipped_tiles[:10])
        audit_lines.append(f"  e.g. {preview}{' …' if len(skipped_tiles) > 10 else ''}")
    else:
        audit_lines.append("Skipped (no/invalid metadata): 0")

    w_px_m = int(np.median([w for w, _ in tile_sizes])) if tile_sizes else 0
    h_px_m = int(np.median([h for _, h in tile_sizes])) if tile_sizes else 0
    step_um_x = (np.median(step_um_x_candidates) if step_um_x_candidates else w_px_m * um_per_px)
    step_um_y = (np.median(step_um_y_candidates) if step_um_y_candidates else h_px_m * um_per_px)

    ix = np.rint(xs / max(step_um_x, 1e-9)).astype(int)
    iy = np.rint(ys / max(step_um_y, 1e-9)).astype(int)
    occ: Dict[Tuple[int, int], List[int]] = {}
    for k in range(len(names)):
        occ.setdefault((ix[k], iy[k]), []).append(k)

    min_ix, max_ix = int(ix.min()), int(ix.max())
    min_iy, max_iy = int(iy.min()), int(iy.max())
    ncols = max_ix - min_ix + 1
    nrows = max_iy - min_iy + 1
    expected_order = [(i, j) for j in range(min_iy, max_iy + 1) for i in range(min_ix, max_ix + 1)]
    present = set(occ.keys())
    missing_cells = [c for c in expected_order if c not in present]
    duplicates = [(cell, idxs) for cell, idxs in occ.items() if len(idxs) > 1]

    audit_lines.append("Grid inference:")
    audit_lines.append(f"  step_um ~ ({step_um_x:.3f}, {step_um_y:.3f}), tile_px ~ ({w_px_m},{h_px_m})")
    audit_lines.append(f"  grid cols×rows ~ {ncols}×{nrows} → expected {ncols*nrows} cells")
    audit_lines.append(f"  present {len(present)}, missing {len(missing_cells)}, duplicates {len(duplicates)}")

    def cell_name(cell: Tuple[int,int]) -> str | None:
        idxs = occ.get(cell)
        return names[idxs[0]] if idxs else None

    if missing_cells:
        audit_lines.append("Missing cells (first 20 shown):")
        for cell in missing_cells[:20]:
            cx, cy = cell
            x_um_miss = cx * step_um_x; y_um_miss = cy * step_um_y
            x_px_miss = x_um_miss / um_per_px; y_px_miss = y_um_miss / um_per_px
            pos = expected_order.index(cell)
            prev_name = next_name = None
            for k in range(pos - 1, -1, -1):
                nm = cell_name(expected_order[k])
                if nm: prev_name = nm; break
            for k in range(pos + 1, len(expected_order)):
                nm = cell_name(expected_order[k])
                if nm: next_name = nm; break
            audit_lines.append(
                f"  cell ({cx},{cy}) ~ stage (µm)=({x_um_miss:.1f},{y_um_miss:.1f}) "
                f"~ pixel (px)=({x_px_miss:.1f},{y_px_miss:.1f})  "
                f"between: prev={prev_name or '—'}, next={next_name or '—'}"
            )

    if duplicates:
        audit_lines.append("Duplicate cells (grid index → tiles):")
        for (cell, idxs) in duplicates[:10]:
            audit_lines.append(f"  {cell}: " + ", ".join(names[i] for i in idxs))

    print("\n[Tile audit]")
    for line in audit_lines[:12]:
        print(line)
    if len(audit_lines) > 12:
        print("  … see full report in tiles_audit.txt")
    (tiles_dir / "tiles_audit.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    # ---------------------- Corner sanity ----------------------
    xn = xs / (xs.max() if xs.max() > 0 else 1.0)
    yn = ys / (ys.max() if ys.max() > 0 else 1.0)
    tl = int(np.argmin(xn + yn))
    tr = int(np.argmin((1 - xn) + yn))
    bl = int(np.argmin(xn + (1 - yn)))
    br = int(np.argmin((1 - xn) + (1 - yn)))
    print("\n[Corner sanity]")
    print(f"  top-left     : {names[tl]}")
    print(f"  top-right    : {names[tr]}")
    print(f"  bottom-left  : {names[bl]}")
    print(f"  bottom-right : {names[br]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
