import os.path
from os.path import splitext, join as pjoin
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Define a class that contains methods to handle all the necessary steps in chunking up a larger wav file,
# saving these chunks, reading them and then creating spectrograms of these.
# I am making the decision here that I am unlikely to ever need the wav files or wav chunks in their original form
# and not as spectrograms.
class WavProcessor:
    def __init__(self, file_path, chunk_duration=3):
        # These create the filepath, sample rate and data from the wav file, and root name of the file
        self.chunk_duration = chunk_duration
        self.file_path = file_path
        self.sample_rate, self.data = wavfile.read(file_path)
        self.base_name = os.path.splitext(os.path.basename(file_path))[0]

    def split_into_chunks(self):
        chunk_size = int(self.chunk_duration * self.sample_rate)
        num_chunks = len(self.data) // chunk_size
        chunks = [self.data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        return chunks

    def save_chunks(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        chunks = self.split_into_chunks()

        for i, chunk in enumerate(chunks):
            out_filename = f"{self.base_name}_chunk_{i:03d}.wav"
            out_path = os.path.join(output_dir, out_filename)
            wavfile.write(out_path, self.sample_rate, chunk)
            print(f"Saved: {out_filename}")

    def save_spectrogram(self, chunk_dir, spec_output_dir):
        os.makedirs(spec_output_dir, exist_ok=True)
        wav_files = [f for f in os.listdir(chunk_dir) if f.lower().endswith(".wav")]

        for i, filename in enumerate(wav_files):
            chunk_path = os.path.join(chunk_dir, filename)
            chunk_sample_rate, chunk_data = wavfile.read(chunk_path)
            spec_basename = os.path.splitext(os.path.basename(filename))[0]
            spec_out_filename = f"{spec_basename}.jpeg"
            spec_output_path = os.path.join(spec_output_dir, spec_out_filename)
            plt.figure(figsize=(10, 4))
            plt.specgram(chunk_data, Fs=chunk_sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
            plt.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()
            plt.savefig(spec_output_path)
            plt.close()
