#!/usr/bin/env python3
import os
import json
import argparse
import math
import sys
from glob import glob
from statistics import mean, median, pstdev


def load_tile_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def window_sum(spectrum, start, end):
    n = len(spectrum)
    if n == 0:
        return 0.0
    s = max(0, min(start, n - 1))
    e = max(0, min(end,   n - 1))
    if e < s:
        s, e = e, s
    return float(sum(spectrum[s:e + 1]))


def percentile(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    i = (len(v) - 1) * q / 100.0
    lo = math.floor(i)
    hi = math.ceil(i)
    if lo == hi:
        return v[int(i)]
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def robust_min_max(values, lo_pct=1.0, hi_pct=99.0):
    if not values:
        return (0.0, 1.0)
    lo = percentile(values, lo_pct)
    hi = percentile(values, hi_pct)
    if hi <= lo:
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def hex_to_rgb(hex_color):
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    r, g, b = [int(round(clamp(c, 0, 255))) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_rgb(c1, c2, t):
    return tuple(lerp(c1[i], c2[i], t) for i in range(3))


PALETTES = {
    "red": [
        (0.00, "#fff5f0"),
        (0.20, "#fcbba1"),
        (0.40, "#fc9272"),
        (0.60, "#fb6a4a"),
        (0.80, "#de2d26"),
        (1.00, "#a50f15"),
    ],
    "blue": [
        (0.00, "#f7fbff"),
        (0.20, "#c6dbef"),
        (0.40, "#9ecae1"),
        (0.60, "#6baed6"),
        (0.80, "#3182bd"),
        (1.00, "#08519c"),
    ],
    "viridis": [
        (0.00, "#440154"),
        (0.25, "#3b528b"),
        (0.50, "#21918c"),
        (0.75, "#5ec962"),
        (1.00, "#fde725"),
    ],
    "inferno": [
        (0.00, "#000004"),
        (0.25, "#57106e"),
        (0.50, "#bc3754"),
        (0.75, "#f98e09"),
        (1.00, "#fcffa4"),
    ],
    "turbo": [
        (0.00, "#30123b"),
        (0.20, "#4145ab"),
        (0.40, "#2ea8df"),
        (0.60, "#7fd34e"),
        (0.80, "#f9ba38"),
        (1.00, "#d93806"),
    ],
}


def color_from_palette(name, t):
    t = clamp(t, 0.0, 1.0)
    stops = PALETTES[name]
    rgb_stops = [(p, hex_to_rgb(c)) for p, c in stops]

    if t <= rgb_stops[0][0]:
        return rgb_to_hex(rgb_stops[0][1])
    if t >= rgb_stops[-1][0]:
        return rgb_to_hex(rgb_stops[-1][1])

    for i in range(len(rgb_stops) - 1):
        p0, c0 = rgb_stops[i]
        p1, c1 = rgb_stops[i + 1]
        if p0 <= t <= p1:
            local_t = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return rgb_to_hex(lerp_rgb(c0, c1, local_t))

    return rgb_to_hex(rgb_stops[-1][1])


def valid_entry(e):
    required = [
        "rownum",
        "colnum",
        "X_rel_um",
        "Y_rel_um",
        "TileWidth_um",
        "TileHeight_um",
        "aggregatedspectrum",
    ]
    return isinstance(e, dict) and all(k in e for k in required)


def build_stats(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stddev_population": None,
            "percentiles": {},
        }

    percentiles = {
        "p01": percentile(values, 1),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "stddev_population": pstdev(values) if len(values) > 1 else 0.0,
        "percentiles": percentiles,
    }


def print_stats(stats, band_start, band_end):
    print(f"Band {band_start}–{band_end}")
    print(f"N     = {stats['count']}")
    print(f"min   = {stats['min']:.6f}")
    print(f"p01   = {stats['percentiles']['p01']:.6f}")
    print(f"p05   = {stats['percentiles']['p05']:.6f}")
    print(f"p10   = {stats['percentiles']['p10']:.6f}")
    print(f"p25   = {stats['percentiles']['p25']:.6f}")
    print(f"p50   = {stats['percentiles']['p50']:.6f}")
    print(f"p75   = {stats['percentiles']['p75']:.6f}")
    print(f"p90   = {stats['percentiles']['p90']:.6f}")
    print(f"p95   = {stats['percentiles']['p95']:.6f}")
    print(f"p99   = {stats['percentiles']['p99']:.6f}")
    print(f"max   = {stats['max']:.6f}")
    print(f"mean  = {stats['mean']:.6f}")
    print(f"med   = {stats['median']:.6f}")
    print(f"std   = {stats['stddev_population']:.6f}")
    print()
    print("Suggested robust scaling:")
    print(f"  --vmin {stats['percentiles']['p05']:.6f} --vmax {stats['percentiles']['p95']:.6f}")
    print("Suggested stronger clipping:")
    print(f"  --vmin {stats['percentiles']['p10']:.6f} --vmax {stats['percentiles']['p90']:.6f}")


def save_histogram(values, out_path, bins, band_start, band_end):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for --histogram-out but is not installed."
        ) from exc

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.xlabel(f"Raw band sum ({band_start}–{band_end})")
    plt.ylabel("Count")
    plt.title(f"Band-sum histogram: {band_start}–{band_end}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build microns-based quadrant heatmap rectangles from per-tile JSONs "
            "(with srcJson for spectra tooltips)."
        )
    )

    # Required scientific inputs
    ap.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing per-tile JSON files (e.g. aggregated-spectra_3x3_json)"
    )
    ap.add_argument(
        "--band-start",
        type=int,
        required=True,
        help="Start channel of the band (inclusive)"
    )
    ap.add_argument(
        "--band-end",
        type=int,
        required=True,
        help="End channel of the band (inclusive)"
    )

    # Output
    ap.add_argument(
        "--output",
        help="Output JSON (default: band_<start>_<end>_heatmap.json in input folder)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write an overlay JSON; only analyze/report"
    )

    # Reporting / histogram
    ap.add_argument(
        "--report",
        action="store_true",
        help="Print descriptive statistics for raw band-sum values"
    )
    ap.add_argument(
        "--report-json",
        help="Optional path to save band-sum statistics as JSON"
    )
    ap.add_argument(
        "--histogram-out",
        help="Optional path to save a histogram PNG of raw band-sum values"
    )
    ap.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Number of bins for histogram output (default 50)"
    )

    # Scaling and visibility
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="Gamma for contrast mapping (default 0.6)"
    )
    ap.add_argument(
        "--robust",
        action="store_true",
        help="Use robust percentile min/max when vmin/vmax are not given"
    )
    ap.add_argument(
        "--robust-lo",
        type=float,
        default=1.0,
        help="Low percentile for robust mode (default 1)"
    )
    ap.add_argument(
        "--robust-hi",
        type=float,
        default=99.0,
        help="High percentile for robust mode (default 99)"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Raw band-sum threshold below which rectangles become fully transparent"
    )
    ap.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Manual lower bound for normalization in raw band-sum units"
    )
    ap.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Manual upper bound for normalization in raw band-sum units"
    )
    ap.add_argument(
        "--opacity-min",
        type=float,
        default=0.05,
        help="Minimum opacity for visible rectangles after normalization"
    )
    ap.add_argument(
        "--opacity-max",
        type=float,
        default=0.85,
        help="Maximum opacity for visible rectangles after normalization"
    )

    # Appearance
    ap.add_argument(
        "--fill",
        default="#ff0000",
        help="Single fill color when no palette is used"
    )
    ap.add_argument(
        "--palette",
        choices=["none", "red", "blue", "viridis", "inferno", "turbo"],
        default="none",
        help="Use a per-rectangle color palette instead of a single fill color"
    )
    ap.add_argument(
        "--stroke",
        default="#000000",
        help="Outline color (default black)"
    )
    ap.add_argument(
        "--stroke-px",
        type=float,
        default=1.5,
        help="Outline width in on-screen pixels (default 1.5)"
    )
    ap.add_argument(
        "--keep-raw-intensity",
        action="store_true",
        help="Keep intensity_raw in the output JSON for debugging"
    )

    args = ap.parse_args()

    if args.robust_lo < 0 or args.robust_hi > 100 or args.robust_hi <= args.robust_lo:
        raise SystemExit("Invalid robust percentile range")
    if args.opacity_min < 0 or args.opacity_min > 1 or args.opacity_max < 0 or args.opacity_max > 1:
        raise SystemExit("opacity-min and opacity-max must be within [0, 1]")
    if args.opacity_max < args.opacity_min:
        raise SystemExit("opacity-max must be >= opacity-min")
    if args.vmin is not None and args.vmax is not None and args.vmax <= args.vmin:
        raise SystemExit("vmax must be greater than vmin")
    if args.hist_bins <= 0:
        raise SystemExit("hist-bins must be > 0")

    in_dir = os.path.abspath(args.input_folder)
    out_path = args.output or os.path.join(
        in_dir,
        f"band_{args.band_start}_{args.band_end}_heatmap.json"
    )

    files = sorted(
        p for p in glob(os.path.join(in_dir, "*.json"))
        if os.path.abspath(p) != os.path.abspath(out_path)
    )
    if not files:
        raise SystemExit(f"No JSON files found in {in_dir}")

    rectangles = []
    sums = []
    folder_name = os.path.basename(os.path.normpath(in_dir))

    skipped_files = 0
    skipped_entries = 0

    for jf in files:
        try:
            entries = load_tile_entries(jf)
        except Exception as exc:
            print(f"Warning: could not read {jf}: {exc}", file=sys.stderr)
            skipped_files += 1
            continue

        if not isinstance(entries, list) or not entries:
            skipped_files += 1
            continue

        valid_entries = [e for e in entries if valid_entry(e)]
        if not valid_entries:
            skipped_files += 1
            continue

        src_json_rel = f"{folder_name}/{os.path.basename(jf)}"

        try:
            max_col = max(int(e["colnum"]) for e in valid_entries)
            max_row = max(int(e["rownum"]) for e in valid_entries)
        except Exception:
            skipped_files += 1
            continue

        cols = max_col + 1
        rows = max_row + 1

        for e in valid_entries:
            try:
                base_x = float(e["X_rel_um"])
                base_y = float(e["Y_rel_um"])
                tile_w = float(e["TileWidth_um"])
                tile_h = float(e["TileHeight_um"])
                c = int(e["colnum"])
                r = int(e["rownum"])
                spec = e["aggregatedspectrum"]

                if not isinstance(spec, list):
                    skipped_entries += 1
                    continue

                qw = float(e.get("quad_width_um", tile_w / cols))
                qh = float(e.get("quad_height_um", tile_h / rows))
                qx = base_x + float(e.get("quad_x_um", c * (tile_w / cols)))
                qy = base_y + float(e.get("quad_y_um", r * (tile_h / rows)))

                raw_sum = window_sum(spec, args.band_start, args.band_end)
                sums.append(raw_sum)

                rect = {
                    "type": "rect",
                    "units": "microns",
                    "x": qx,
                    "y": qy,
                    "width": qw,
                    "height": qh,
                    "fill": args.fill,
                    "stroke": args.stroke,
                    "strokeWidthPx": args.stroke_px,
                    "strokeWidth": 0.0005,
                    "bandStart": args.band_start,
                    "bandEnd": args.band_end,
                    "bandValue": round(raw_sum, 6),
                    "intensity_raw": raw_sum,
                    "basename": e.get("basename", ""),
                    "colnum": c,
                    "rownum": r,
                    "srcJson": src_json_rel,
                    "label": f'{e.get("basename", "")} · r{r}, c{c}',
                }
                rectangles.append(rect)

            except Exception:
                skipped_entries += 1
                continue

    if not rectangles:
        raise SystemExit("No valid quadrant entries found")

    stats = build_stats(sums)

    if args.report:
        print_stats(stats, args.band_start, args.band_end)

    if args.report_json:
        report_payload = {
            "input_folder": in_dir,
            "band_start": args.band_start,
            "band_end": args.band_end,
            "stats": stats,
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Wrote stats JSON → {args.report_json}")

    if args.histogram_out:
        save_histogram(sums, args.histogram_out, args.hist_bins, args.band_start, args.band_end)
        print(f"Wrote histogram PNG → {args.histogram_out}")

    if args.dry_run:
        print("Dry run requested; no overlay JSON written.")
        if skipped_files or skipped_entries:
            print(
                f"Skipped {skipped_files} files and {skipped_entries} entries that did not match the expected format",
                file=sys.stderr
            )
        return

    # Choose normalization bounds
    if args.vmin is not None or args.vmax is not None:
        lo = args.vmin if args.vmin is not None else min(sums)
        hi = args.vmax if args.vmax is not None else max(sums)
        if hi <= lo:
            hi = lo + 1.0
    elif args.robust:
        lo, hi = robust_min_max(sums, args.robust_lo, args.robust_hi)
    else:
        lo, hi = min(sums), max(sums)
        if hi <= lo:
            hi = lo + 1.0

    norm_lo = max(lo, args.threshold) if args.threshold is not None else lo
    norm_hi = hi
    if norm_hi <= norm_lo:
        norm_hi = norm_lo + 1.0

    norm_rng = norm_hi - norm_lo

    for rect in rectangles:
        raw = rect["intensity_raw"]
        hidden = args.threshold is not None and raw < args.threshold

        if hidden:
            intensity = 0.0
            opacity = 0.0
        else:
            v = (raw - norm_lo) / norm_rng
            v = clamp(v, 0.0, 1.0)
            if args.gamma and args.gamma > 0:
                v = v ** args.gamma
            intensity = v
            opacity = args.opacity_min + v * (args.opacity_max - args.opacity_min)

        rect["intensity"] = round(intensity, 6)
        rect["fillOpacity"] = round(opacity, 6)

        if args.palette != "none":
            rect["fill"] = color_from_palette(args.palette, intensity)

        if not args.keep_raw_intensity:
            rect.pop("intensity_raw", None)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rectangles, f, indent=2)

    print(f"Wrote {len(rectangles)} rectangles → {out_path}")
    if skipped_files or skipped_entries:
        print(
            f"Skipped {skipped_files} files and {skipped_entries} entries that did not match the expected format",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()