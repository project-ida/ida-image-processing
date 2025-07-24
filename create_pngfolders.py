import tifffile
from pathlib import Path
from PIL import Image
from collections import defaultdict
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Extract PNGs from OME-TIFF Z-stacks.")
parser.add_argument('--folder', required=True, help="Folder containing .ome.tif files.")
args = parser.parse_args()

# Path to your folder with .ome.tif files
input_folder = Path(args.folder)
ome_files = sorted(input_folder.glob("*.ome.tif"))

if not ome_files:
    raise FileNotFoundError(f"No .ome.tif files found in folder: {input_folder}")

ome_file = ome_files[0]

# Read metadata and get all pages
with tifffile.TiffFile(ome_file) as tif:
    series = tif.series[0]
    pages = series.pages
    total_pages = len(pages)

    # Try to get the file source name, skip if not available
    file_sources = []
    valid_pages = []
    for p in pages:
        try:
            stem = Path(p.parent.filehandle.name).stem
            file_sources.append(stem)
            valid_pages.append(p)
        except Exception as e:
            print(f"⚠️ Skipping page due to error: {e}")
            continue

    if not valid_pages:
        raise RuntimeError("❌ No valid pages to process.")

    unique_files = list(dict.fromkeys(file_sources))
    num_tiles = len(unique_files)
    num_z = len(valid_pages) // num_tiles if num_tiles > 0 else 1
    print(f"🧠 Found {num_tiles} unique tiles with {num_z} Z-slices each")

    # Group pages by Z-layer index
    z_layers = defaultdict(list)
    for i, (page, fname) in enumerate(zip(valid_pages, file_sources)):
        z_index = i % num_z
        z_layers[z_index].append((page, fname))

    # Save to per-Z output folders
    for z, entries in z_layers.items():
        output_folder = input_folder / f"png_output_z{z}"
        output_folder.mkdir(exist_ok=True)

        for page, base_name in entries:
            try:
                img = page.asarray()
                png_path = output_folder / f"{base_name}.png"
                Image.fromarray(img).save(png_path, "PNG")
                print(f"✅ Saved Z{z}: {png_path.name}")
            except Exception as e:
                print(f"⚠️ Could not save image {base_name}.png: {e}")
