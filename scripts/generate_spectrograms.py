from src.data_processing.spectrogram_pipeline import spectrogramPipeline
from src.data_processing.wav_processor import WavProcessor
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

labelled_wav_path = PROJECT_ROOT / "data" / "processed" / "labelled_chunks_3s"
labelled_spec_path = PROJECT_ROOT / "data" / "processed" / "spectrogram_3s"

labelled_wav_files = list(labelled_wav_path.glob("*.wav"))


