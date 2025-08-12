import re
from src.data_processing.spectrogram_pipeline import spectrogramPipeline
from src.data_processing.wav_processor import WavProcessor
from pathlib import Path
import pandas as pd
import os
from os.path import splitext, basename

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
# Location to source labelled chunks from
labelled_wav_path = PROJECT_ROOT / "data" / "processed" / "labelled_chunks_3s"
# Location to save spectrograms to
labelled_spec_path = PROJECT_ROOT / "data" / "processed" / "spectrogram_3s"
# We will also be creating a csv to hold the filenames and labels
data_path = PROJECT_ROOT / "src" / "data" / "annotations"
# Ensure output directories exist
labelled_spec_path.mkdir(parents=True, exist_ok=True)
data_path.mkdir(parents=True, exist_ok=True)

labelled_wav_files = list(labelled_wav_path.glob("*.wav"))

labels_and_filenames = pd.DataFrame(columns=['filename', 'label'])
for wav in labelled_wav_files:
    wav_proc = WavProcessor(wav)
    spec_proc = spectrogramPipeline(wav_proc)

    mel_spec = spec_proc.fft_and_mel(win_len=512)
    scaled_spec = spec_proc.scaling(spec=mel_spec, rate=wav_proc.sample_rate, hop_length=384)
    final_spec = spec_proc.clip_and_normalise(scaled_spec)
    filename = splitext(basename(wav))[0]
    print(f'Successfully created {filename}')
    parts = re.split(r'k_\d+_', filename)
    if len(parts) > 1:
        label = parts[1]
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    labels_and_filenames.loc[len(labels_and_filenames)] = [filename, label]
    output_path = labelled_spec_path / f"{filename}.png"
    spec_proc.save_spectrogram_image(final_spec, output_path)
    print(f'Successfully saved {filename}')

labels_and_filenames.to_csv(os.path.join(data_path, 'spectrogram_labels.csv'), index=False)

