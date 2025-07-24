import os
import sys
import glob
import argparse
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate TileConfiguration.txt from Micro-Manager metadata.")
parser.add_argument('--folder', required=True, help="Folder containing metadata and image files.")
args = parser.parse_args()

folder = args.folder

if not os.path.isdir(folder):
    print(f"❌ Error: '{folder}' is not a valid directory.")
    sys.exit(1)

# Output file
output_file = "TileConfiguration.txt"

# Collect entries: (filename, x_um, y_um, pixel_size_um)
entries = []

# Scan all *_metadata.txt files
for filepath in sorted(glob.glob(os.path.join(folder, "*_metadata.txt"))):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"Processing file: {filepath}")

            # Extract values using string matching
            x_match = re.search(r'"XPositionUm":\s*(\d+\.\d+)', content)
            y_match = re.search(r'"YPositionUm":\s*(\d+\.\d+)', content)
            pixel_size_match = re.search(r'"PixelSizeUm":\s*(\d+\.\d+)', content)
            filename_match = re.search(r'"FileName":\s*"([^"]+)"', content)

            x = float(x_match.group(1)) if x_match else None
            y = float(y_match.group(1)) if y_match else None
            pixel_size = float(pixel_size_match.group(1)) if pixel_size_match else None
            filename = filename_match.group(1) if filename_match else None

            print(f"Extracted: x={x}, y={y}, pixel_size={pixel_size}, filename={filename}")

            if x is not None and y is not None and pixel_size is not None and filename is not None and pixel_size != 0:
                base_name = os.path.splitext(filename)[0]
                jpg_filename = base_name + ".jpg"
                entries.append((jpg_filename, x, y, pixel_size))
                print(f"Added to entries: {jpg_filename}, x={x}, y={y}, pixel_size={pixel_size}")
            else:
                print(f"Skipping invalid entry from {filepath}: x={x}, y={y}, pixel_size={pixel_size}, filename={filename}")

    except Exception as e:
        print(f"Warning: Error processing {filepath}: {e} - Skipping file.")

if not entries:
    raise ValueError("No valid entries found in metadata files.")

# Normalize to set origin at (0, 0)
try:
    x0 = min(e[1] for e in entries)
    y0 = min(e[2] for e in entries)
except ValueError as e:
    raise ValueError("Failed to determine minimum coordinates. Check if entries contain valid x and y values.") from e

# Write TileConfiguration.txt
try:
    with open(output_file, "w") as f:
        f.write("dim = 2\n")
        for filename, x_um, y_um, pixel_size_um in entries:
            try:
                x_px = (x_um - x0) / pixel_size_um
                y_px = -1 * (y_um - y0) / pixel_size_um
                f.write(f"{filename}; ; ({x_px:.1f}, {y_px:.1f})\n")
            except ZeroDivisionError:
                print(f"Skipping entry {filename} due to zero pixel_size_um")
            except Exception as e:
                print(f"Error processing entry {filename}: {e}")
    print(f"✅ {output_file} created with {len(entries)} tiles.")
except IOError as e:
    print(f"❌ Error writing to {output_file}: {e}")
    sys.exit(1)
