from pathlib import Path
import os.path
import numpy as np
from scipy.io import wavfile


def save_npy_array_first_time(chunk_dir, npy_output_dir):
    os.makedirs(npy_output_dir, exist_ok=True)
    wav_files = [f for f in os.listdir(chunk_dir) if f.lower().endswith(".wav")]

    for i, filename in enumerate(wav_files):
        chunk_path = os.path.join(chunk_dir, filename)
        chunk_sample_rate, chunk_data = wavfile.read(chunk_path)
        # Check in case data is stereo. I don't think it is but this just in case.
        if len(chunk_data.shape) == 2:
            chunk_data = chunk_data.mean(axis=1)  # Convert to mono
        if chunk_data.dtype == np.int16:
            chunk_data = chunk_data.astype(np.float32) / 32768.0
        npy_basename = os.path.splitext(os.path.basename(filename))[0]
        npy_out_filename = f"{npy_basename}.npy"
        npy_output_path = os.path.join(npy_output_dir, npy_out_filename)
        np.save(npy_output_path, chunk_data)
        print(f"Saved {i + 1}/{len(wav_files)}: {npy_out_filename}")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

chunk_files = PROJECT_ROOT / "data" / "processed" / "chunks_3s"
npy_path = PROJECT_ROOT / "data" / "processed" / "arrays_3s"

save_npy_array_first_time(chunk_files, npy_path)