import os.path
from os.path import splitext, join as pjoin
from scipy.io import wavfile


file_path = pjoin("data", "raw", "20240611_050000.WAV")


def split_wav_file(input_path, output_dir, chunk_length=3.0):
    # Read file
    sample_rate, data = wavfile.read(input_path)
    total_samples = data.shape[0]
    chunk_samples = int(chunk_length * sample_rate)
    num_chunks = total_samples // chunk_samples
    # The numbers in the np array of data correspond to amplitude values at a moment in time

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = data[start:end]

        # Output path
        out_filename = f"{base_name}_chunk_{i:03d}.wav"
        out_path = os.path.join(output_dir, out_filename)

        # Save using wavfile.write
        wavfile.write(out_path, sample_rate, chunk)

        print(f"Saved: {out_filename}")


# of note - data is mono (data.ndim = 1)
split_wav_file(file_path, "data/processed/chunks_3s")