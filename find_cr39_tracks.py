"""
find_tracks.py

This script processes microscope image tiles of CR39 detector plates to detect 
particle tracks (pits), extract geometric features, optionally store them in a 
PostgreSQL database, and overwrite the original images with visual overlays 
marking the detected tracks.

Expected image layout:
    data/<experiment_id>/<zoom_level>/<X>_<Y>.png
Where:
    <experiment_id> : Provided as a command-line argument.
    <zoom_level>    : Matches the ZOOM_LEVEL constant in this script.
    <X>, <Y>        : Integer tile coordinates (used for tile_position in DB).

Features extracted for each track:
    - Diameter
    - Major axis length
    - Minor axis length
    - Circularity
    - Lightest pixel coordinates (x_light, y_light)
    - Darkest pixel coordinates (x_dark, y_dark)

Overlay output:
    - Green ellipse or circle drawn around each detected pit.
    - Original image is overwritten with overlay applied.

Usage:
    python script.py <experiment_id>

Configuration (constants at top of file):
    DATA_FOLDER : Root directory for data (default: "data")
    ZOOM_LEVEL  : Zoom level subfolder to process
    MIN_AREA    : Minimum pit area in pixels
    SAVE_TO_DB  : Set to True to insert results into PostgreSQL
"""


import os
import sys
import cv2
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from psql_credentials import PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

# === CONFIGURATION ===
DATA_FOLDER = "data"
ZOOM_LEVEL = 4
MIN_AREA = 1  # pixels
SAVE_TO_DB = True

# === DATABASE CONNECTION ===
def get_connection():
    conn = psycopg2.connect(
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        host=PGHOST,
        port=PGPORT,
        connect_timeout=10
    )
    return conn

# === FEATURE EXTRACTION ===
def extract_features(img, min_area=50):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    overlays = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w / 2, y + h / 2
        perim = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
        diameter = np.sqrt(4 * area / np.pi)

        if len(cnt) >= 5:
            (_, _), (MA, ma), angle = cv2.fitEllipse(cnt)
            major = max(MA, ma)
            minor = min(MA, ma)
        else:
            major = minor = 0
            angle = 0

        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
        ys, xs = np.where(mask == 1)
        vals = img[ys, xs]
        idx_min, idx_max = np.argmin(vals), np.argmax(vals)
        x_dark, y_dark = int(xs[idx_min]), int(ys[idx_min])
        x_light, y_light = int(xs[idx_max]), int(ys[idx_max])

        features = {
            'diameter': round(float(diameter),4),
            'major': round(float(major),4),
            'minor': round(float(minor),4),
            'circularity': round(float(circularity),4),
            'x_light': x_light,
            'y_light': y_light,
            'x_dark': x_dark,
            'y_dark': y_dark
        }

        results.append({
            'track_position': [float(cx), float(cy)],
            'features': features
        })

        overlays.append((cx, cy, diameter, major, minor, circularity,
                         x_light, y_light, x_dark, y_dark, angle))

    return results, overlays

# === OVERLAY SAVING ===
def save_overlay(img, overlays, out_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    ax = plt.gca()

    for cx, cy, diameter, major, minor, circularity, x_light, y_light, x_dark, y_dark, angle in overlays:
        if circularity >= 0.8 or (major == 0 and minor == 0):
            r = diameter / 2
            ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor='green', linewidth=3))
        else:
            ax.add_patch(Ellipse((cx, cy), width=major, height=minor, angle=angle,
                                 fill=False, edgecolor='green', linewidth=3))

        # ax.plot(x_light, y_light, 'bo', markersize=4)
        # ax.plot(x_dark,  y_dark,  'ro', markersize=4)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

# === MAIN ===
def main():
    if len(sys.argv) != 2:
        print("ERROR: Missing argument `python script.py <experiment_id>`")
        sys.exit(1)

    experiment_id = sys.argv[1]
    input_folder = os.path.join(DATA_FOLDER, experiment_id, str(ZOOM_LEVEL))

    if not os.path.isdir(input_folder):
        print(f"Folder not found: {input_folder}")
        sys.exit(1)

    conn = get_connection()
    cur = conn.cursor()

    num_files_processed = 0
    print(f"Processing {experiment_id} at zoom level {ZOOM_LEVEL}")
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".png"):
            continue
        try:
            num_files_processed += 1
            print(f"Processing {filename}")
            tile_x, tile_y = map(int, filename.rsplit(".", 1)[0].split("_"))
        except ValueError:
            print(f"Skipping {filename}: invalid name format")
            continue

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        results, overlays = extract_features(img, min_area=MIN_AREA)

        if SAVE_TO_DB == True:
            for res in results:
                cur.execute("""
                    INSERT INTO cr39_tracks (experiment_id, tile_position, track_position, features)
                    VALUES (%s, %s, %s, %s)
                """, (
                    experiment_id,                      # text
                    [tile_x, tile_y],                    # integer[]
                    [res['track_position'][0], res['track_position'][1]],  # double precision[]
                    [
                        float(res['features']['minor']),
                        float(res['features']['major'])
                    ]  # numeric[]
                ))


        # Overwrite image with overlay
        save_overlay(img, overlays, img_path)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Finished processing {num_files_processed} files.")

if __name__ == "__main__":
    main()
