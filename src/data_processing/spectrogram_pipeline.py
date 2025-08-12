import numpy as np
import librosa
from librosa.filters import mel
from scipy.signal import butter, sosfiltfilt, spectrogram, windows, lfilter
from PIL import Image
from pathlib import Path

class spectrogramPipeline:
    # This requires the WavProcessor class to be initiated.
    def __init__(self, wav_processor):
        self.data = wav_processor.data
        self.sample_rate = wav_processor.sample_rate
        self.chunk_duration = wav_processor.chunk_duration
        self.base_name = wav_processor.base_name

    # The following methods are designed to emulate the preprocessing steps taken by BirdNET
    # Optional: Apply bandpass filter
    def bandpass_filter(self, fmin, fmax):
        bandpass = butter(4, [fmin, fmax], btype='bandpass', output='sos', fs=48000)
        filtered = sosfiltfilt(bandpass, self.data)
        return filtered

    def fft_and_mel(self, win_len, shape=None, fmin=150, fmax=15000, apply_filter=False):
        # Define parameters for FFT
        if shape is None:
            shape = [64, 512]
        data = self.data
        n_fft = win_len
        overlap = (win_len // 4)
        # Apply bandpass filter if desired
        if apply_filter:
            data = self.bandpass_filter(fmin, fmax)

        # Create spectrogram
        freqs, times, specs = spectrogram(
            x=data,
            fs=self.sample_rate,
            window=windows.hann(win_len),
            nperseg=win_len,
            noverlap=overlap,
            nfft=n_fft)

        # Convert to mel scale
        mel_filter = mel(sr=self.sample_rate, n_fft=n_fft, n_mels=shape[0], fmin=fmin, fmax=fmax)
        mel_freqs = librosa.mel_frequencies(n_mels=shape[0], fmin=fmin, fmax=fmax)
        mel_spec = mel_filter @ specs

        return mel_spec

    def scaling(self, spec, rate, hop_length, gain=0.8, bias=10, power=0.25, t=0.060, eps=1e-6, scale='log'):
        if scale == 'log':
            mel_scaled = librosa.power_to_db(spec ** 2, ref=np.max, top_db=100)
        elif scale == 'pcen':
            # This is taken directly from the BirdNET codebase
            # It is their means of per-channel energy normalisation
            s = 1 - np.exp(- float(hop_length) / (t * rate))
            M = lfilter([s], [1, s - 1], spec)
            smooth = (eps + M) ** (-gain)
            mel_scaled = (spec * smooth + bias) ** power - bias ** power
        return mel_scaled

    def clip_and_normalise(self, spec, shape=None):
        # Trim to desired size if too large
        if shape is None:
            shape = [64, 512]
        nc_spec = spec[:shape[0], :shape[1]]
        # Normalise to between 0 and 1
        nc_spec -= nc_spec.min()
        if not nc_spec.max() == 0:
            nc_spec /= nc_spec.max()
        else:
            nc_spec = np.clip(nc_spec, 0, 1)
        return nc_spec

    def save_spectrogram_image(self, spec, out_path):
        # Scale to 0–255 and convert to uint8
        spec_uint8 = (spec * 255).astype(np.uint8)

        # Create image from array
        img = Image.fromarray(spec_uint8, mode='L')  # 'L' for grayscale

        # Save image
        img.save(out_path)