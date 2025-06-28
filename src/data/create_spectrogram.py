import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from os.path import splitext, join

def create_spectrogram(input_wav):

    sample_rate, data = wavfile.read(input_wav)

    basename = os.path.splitext(os.path.basename(input_wav))[0]
    output_dir = "data/processed/spectrogram_3s"
    os.makedirs(output_dir, exist_ok=True)
    specgram_filename = f"{basename}.jpeg"
    output_path = os.path.join(output_dir, specgram_filename)

    plt.figure(figsize=(10, 4))
    plt.specgram(data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
