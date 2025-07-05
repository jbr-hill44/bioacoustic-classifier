from data_processing.wav_processor import WavProcessor
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

raw_files = PROJECT_ROOT / "data" / "raw"
chunk_path = PROJECT_ROOT / "data" / "processed" / "chunks_3s"
spec_path = PROJECT_ROOT / "data" / "processed" / "spectrogram_3s"

raw_files = list(raw_files.glob("*.WAV"))

for file in raw_files:
    processor = WavProcessor(file, chunk_duration=3)
    processor.save_chunks(chunk_path)
    processor.save_spectrogram(chunk_path, spec_path)


