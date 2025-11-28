#!/usr/bin/env python3
import argparse, json, os, re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from ndstorage import Dataset

# one-time: pip install ndstorage pillow tqdm numpy
# example: python export_ndtiff_tiles.py "C:/Users/MITLENR/leica-images/test-fm/test2full3z_1" --format bmp --overwrite false


def parse_args():
    ap = argparse.ArgumentParser(
        description="Export NDTIFF tiles to per-position, per-Z images + JSON sidecars."
    )
    ap.add_argument("ndtiff_folder", help="Path to the NDTIFF dataset directory")
    ap.add_argument("--out-subdir", default="export",
                    help='Name of output subfolder inside the dataset (default: "export")')
    ap.add_argument("--format", choices=["bmp", "png"], default="bmp",
                    help="Image format for tiles (bmp=8-bit only, png=8-bit lossless)")
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                    help="Overwrite existing files (default: false)")
    return ap.parse_args()


# ---------- metadata helpers ----------
def sanitize(name: str) -> str:
    """Make a safe filename stem from a StagePositions label."""
    return re.sub(r'[^\w.\-]+', "_", name or "Pos")

def parse_mda_slices(summary):
    """Return (offsets_um:list[float], relative:bool, zref:float)."""
    offsets, relative, zref = None, True, 0.0
    # MDA JSON can be under UserData.MDA_Settings.scalar or MdaSettings
    mda = (summary.get("UserData") or {}).get("MDA_Settings", {})
    if isinstance(mda, dict):
        mda = mda.get("scalar")
    candidates = [mda, summary.get("MdaSettings")]
    for c in candidates:
        if isinstance(c, str):
            try:
                jd = json.loads(c)
                offsets  = jd.get("slices", offsets)
                relative = bool(jd.get("relativeZSlice", relative))
                zref     = float(jd.get("zReference", zref))
                if offsets is not None:
                    offsets = [float(x) for x in offsets]
                    break
            except Exception:
                pass
    if offsets is None:
        # Fallback from summary fields
        S = int(summary.get("Slices") or 1)
        step = summary.get("z-step_um")
        if step is None:
            try:
                jd = json.loads(mda) if isinstance(mda, str) else {}
                step = jd.get("sliceZStepUm")
            except Exception:
                step = None
        step = float(step) if step is not None else 0.0
        if S > 1 and step > 0:
            center = (S - 1) / 2.0
            offsets = [(i - center) * step for i in range(S)]
        else:
            offsets = [0.0] * S
    return offsets, bool(relative), float(zref)

def affine_pixel_sizes_um(summary):
    """Extract µm/px from AffineTransform scalar 'ax_bx_by_dy' (assumed present)."""
    aff = summary["UserData"]["AffineTransform"]["scalar"]
    ax, _bx, _by, dy = map(float, aff.split("_"))
    return abs(ax), abs(dy)  # µm/px in X and Y


# ---------- image helpers ----------
def squeeze_leading_singletons(a: np.ndarray) -> np.ndarray:
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    return a

def normalize_to_u8_rgb(a: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Return (array, mode) suitable for PIL Image.fromarray:
    - mono -> (H,W) 'L'
    - RGB  -> (H,W,3) 'RGB'
    - RGBA -> drop alpha
    If uint16, downshift >> 8 to uint8.
    """
    a = squeeze_leading_singletons(a)
    if a.ndim == 2:
        if a.dtype == np.uint16:
            a = (a >> 8).astype(np.uint8)
        else:
            a = a.astype(np.uint8, copy=False)
        return a, "L"

    if a.shape[-1] == 4:  # RGBA -> RGB
        a = a[..., :3]

    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    else:
        a = a.astype(np.uint8, copy=False)

    if a.ndim == 3 and a.shape[-1] == 3:
        return a, "RGB"

    # Fallback: treat last band as L
    if a.ndim == 3 and a.shape[-1] >= 1:
        return a[..., 0].astype(np.uint8, copy=False), "L"

    raise ValueError(f"Unexpected array shape for export: {a.shape}, dtype={a.dtype}")


def main():
    args = parse_args()

    ndpath = Path(args.ndtiff_folder)
    if not ndpath.is_dir():
        raise SystemExit(f"Folder not found: {ndpath}")

    # Open dataset
    ds = Dataset(str(ndpath))
    images = ds.as_array()
    summary = ds.summary_metadata
    stage_positions = summary.get("StagePositions", []) or []
    P = len(stage_positions)
    if P == 0:
        raise SystemExit("No StagePositions in summary_metadata; cannot export.")

    # Dimensions — assume AxisOrder starts with 'position' (your setup)
    try:
        S = int(summary.get("Slices") or (images.shape[3] if images.ndim >= 4 else 1))
    except Exception:
        S = 1

    # Z offsets (absolute Z = base focus Z +/- offset)
    slice_offsets_um, relative_slices, zref_um = parse_mda_slices(summary)

    # Pixel sizes (µm/px) — from AffineTransform per your assumption
    px_x_um, px_y_um = affine_pixel_sizes_um(summary)

    dst = ndpath / args.out_subdir
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Exporting tiles from {ndpath} -> {dst}")
    print(f"Positions: {P}, Z-layers: {S}, format: {args.format.upper()}")

    total = P * S
    idx_flat = 0

    for p in tqdm(range(P), desc="Positions"):
        pos = stage_positions[p]
        label = sanitize(pos.get("Label", f"Pos-{p:04d}"))

        # Pull XY and base Z
        x_um = y_um = z_base_um = None
        for dev in pos.get("DevicePositions", []):
            if dev.get("Device") == pos.get("DefaultXYStage", "XYStage"):
                xy = dev.get("Position_um", [None, None])
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    x_um, y_um = float(xy[0]), float(xy[1])
            if dev.get("Device") == pos.get("DefaultZStage", "FocusDrive"):
                zu = dev.get("Position_um", [None])
                z_base_um = float(zu[0] if isinstance(zu, (list, tuple)) else zu)

        for z in range(S):
            # read one tile (prefer 4D access pos,time,channel,z)
            try:
                a = images[p, 0, 0, z]
            except Exception:
                # fallback to flattened indexing if needed
                a = images[idx_flat]
            try:
                a = a.compute()
            except Exception:
                pass
            a = np.asarray(a)

            arr_u8, mode = normalize_to_u8_rgb(a)

            # filename with Z suffix
            stem = f"{label}_Z{z:02d}"
            img_path = dst / f"{stem}.{args.format}"
            txt_path = dst / f"{stem}.txt"

            if img_path.exists() and not args.overwrite and txt_path.exists():
                idx_flat += 1
                continue

            # save image
            im = Image.fromarray(arr_u8, mode=mode)
            if args.format == "bmp":
                im.save(img_path, format="BMP")
            else:
                im.save(img_path, format="PNG", compress_level=3)

            # compute absolute Z
            dz = float(slice_offsets_um[z]) if z < len(slice_offsets_um) else 0.0
            if relative_slices:
                z_abs_um = (z_base_um if z_base_um is not None else 0.0) + dz
            else:
                z_abs_um = dz  # could add zref_um if your convention needs it

            # sidecar JSON (compact but useful)
            meta = {
                "IndexFlat":   idx_flat,
                "PosIndex":    p,
                "Z_layer":     z,
                "Label":       pos.get("Label", f"Pos-{p:04d}"),
                "X_um":        x_um,
                "Y_um":        y_um,
                "Z_base_um":   z_base_um,
                "Z_offset_um": dz,
                "Z_um":        z_abs_um,
                "PixelSizeX_um_per_px": px_x_um,
                "PixelSizeY_um_per_px": px_y_um,
                "Width_px":    int(arr_u8.shape[1] if arr_u8.ndim >= 2 else 0),
                "Height_px":   int(arr_u8.shape[0] if arr_u8.ndim >= 2 else 0),
                "Mode":        mode,
            }
            with open(txt_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            idx_flat += 1

    print("Done.")


if __name__ == "__main__":
    main()
