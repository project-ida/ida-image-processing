import os
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def process_h5oina_file(filepath, output_folder):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"Processing file: {filepath}")

    with h5py.File(filepath, 'r') as file:
        # --- SEM image ---
        group_path = '/1/Electron Image/Data/SE'
        first_child = list(file[group_path].keys())[0]
        imagedata = file[group_path][first_child][:]

        imagewidth = file['/1/Electron Image/Header/X Cells'][0]
        imageheight = file['/1/Electron Image/Header/Y Cells'][0]
        data_reshaped = imagedata.reshape(imageheight, imagewidth)

        # Save PNG
        fig, ax = plt.subplots(figsize=(imagewidth / 100, imageheight / 100))
        ax.imshow(data_reshaped, cmap='gray')
        plt.axis('off')
        sem_image_path = os.path.join(output_folder, f"{filename}_sem.png")
        fig.savefig(sem_image_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)

        # Save NPZ
        sem_npz_path = os.path.join(output_folder, f"{filename}_sem.npz")
        np.savez_compressed(sem_npz_path, sem_data=data_reshaped)

        # --- EDS spectra ---
        spectrum_data = file['/1/EDS/Data/Spectrum'][:]
        x_pixels = file['/1/EDS/Header/X Cells'][0]
        y_pixels = file['/1/EDS/Header/Y Cells'][0]
        spectrum_length = file['/1/EDS/Header/Number Channels'][0]
        # --- EDS energy calibration from header ---
        channel_width_eV = float(file['/1/EDS/Header/Channel Width'][0])
        start_eV         = float(file['/1/EDS/Header/Start Channel'][0])
        energy_range_eV  = float(file['/1/EDS/Header/Energy Range'][0])
        n_channels       = int(spectrum_length)

        # Derived quantities (nice to have, optional)
        e_min_eV = start_eV
        e_max_eV = start_eV + channel_width_eV * (n_channels - 1)

        spectra_array2 = np.zeros((y_pixels, x_pixels, spectrum_length), dtype=np.uint8)
        chunk_size = 10000
        for start_idx in tqdm(range(0, spectrum_data.shape[0], chunk_size), desc="Processing chunks"):
            end_idx = min(start_idx + chunk_size, spectrum_data.shape[0])
            chunk = spectrum_data[start_idx:end_idx, :]
            for flat_idx, spectrum in enumerate(chunk):
                y_idx = (start_idx + flat_idx) // x_pixels
                x_idx = (start_idx + flat_idx) % x_pixels
                spectra_array2[y_idx, x_idx, :] = spectrum

        eds_npz_path = os.path.join(output_folder, f"{filename}_eds.npz")
        np.savez_compressed(eds_npz_path, eds_data=spectra_array2)

        # --- Metadata ---
        metadata = {
            '/1/Electron Image/Header/X Cells': imagewidth,
            '/1/Electron Image/Header/Y Cells': imageheight,
            '/1/Electron Image/Header/X Step': file['/1/Electron Image/Header/X Step'][0],
            '/1/Electron Image/Header/Y Step': file['/1/Electron Image/Header/Y Step'][0],
            '/1/EDS/Header/X Cells': x_pixels,
            '/1/EDS/Header/Y Cells': y_pixels,
            '/1/EDS/Header/X Step': file['/1/EDS/Header/X Step'][0],
            '/1/EDS/Header/Y Step': file['/1/EDS/Header/Y Step'][0],
            '/1/EDS/Header/Start Channel': start_eV,
            '/1/EDS/Header/Channel Width': channel_width_eV,
            '/1/EDS/Header/Energy Range': energy_range_eV,
            '/1/EDS/Header/Number Channels': n_channels,
            '/1/EDS/Header/Stage Position/X': file['/1/EDS/Header/Stage Position/X'][0],
            '/1/EDS/Header/Stage Position/Y': file['/1/EDS/Header/Stage Position/Y'][0],
            '/1/EDS/Header/Stage Position/Z': file['/1/EDS/Header/Stage Position/Z'][0],

            # --- NEW: viewer-friendly keys for config.txt ---
            'eds_ev_per_ch': channel_width_eV,
            'eds_start_ev':  start_eV,
            'eds_n_channels': n_channels,
            'eds_e_min_ev':  e_min_eV,
            'eds_e_max_ev':  e_max_eV,
        }
        metadata_path = os.path.join(output_folder, f"{filename}_metadata.txt")
        with open(metadata_path, 'w') as meta_file:
            for key, value in metadata.items():
                meta_file.write(f"{key}: {value}\n")

    print(f"Finished processing file: {filename}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert .h5oina files into SEM/EDS npz/png/txt outputs.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing .h5oina files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save outputs")
    parser.add_argument("--skipfiles", type=int, default=0, help="Number of initial files to skip")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    candidates = []
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            if file.endswith(".h5oina") and "Montaged" not in file:
                candidates.append(os.path.join(root, file))

    total = len(candidates)
    if total == 0:
        print(f"No .h5oina files found in: {args.input_folder}")
        return

    if args.skipfiles > 0:
        print(f"Skipping first {args.skipfiles} file(s) out of {total}")

    for idx, filepath in enumerate(candidates, start=1):
        if idx <= args.skipfiles:
            continue
        file = os.path.basename(filepath)
        print(f"Starting processing ({idx} out of {total}) for file: {file}")
        process_h5oina_file(filepath, args.output_folder)

    print("All specified files have been processed.")


if __name__ == "__main__":
    main()
