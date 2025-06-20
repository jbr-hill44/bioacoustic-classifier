from os.path import splitext, join as pjoin
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import wavfile
import scipy.io

file_path = pjoin("data", "raw", "20240611_050000.WAV")


def split_wav_file(input_path,output_dir,chunk_length=3.0):
    # Read file
    sample_rate,data = wavfile.read(input_path)
    total_samples = data.shape[0]
    chunk_samples = int()
    num_chunks = total_samples // chunk_samples
    # The numbers in the np array of data correspond to amplitude values at a moment in time
sample_rate, data = wavfile.read(file_path)
# of note - data is mono (data.ndim = 1)

plt.figure(figsize=(10, 4))
plt.specgram(data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Intensity (dB)")
plt.title("Spectrogram")
plt.tight_layout()
plt.show()