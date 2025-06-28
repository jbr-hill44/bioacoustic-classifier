from data_processing.wav_processor import WavProcessor
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # Adjust this depending on how deep your script is

# Now use the absolute path to your data file
file_path = PROJECT_ROOT / 'data' / 'raw' / '20240611_063000.WAV'
processor = WavProcessor(file_path, chunk_duration=3)

# Step 2: Split and save chunks
chunk_output_dir = PROJECT_ROOT / 'data' / 'processed' / 'chunks_3s'
processor.save_chunks(chunk_output_dir)

# Step 3: Create and save spectrograms of those chunks
spectrogram_output_dir = PROJECT_ROOT / 'data' / 'processed' / 'spectrogram_3s'
processor.save_spectrogram(chunk_output_dir, spectrogram_output_dir)


