from pathlib import Path
import os
from os.path import splitext, basename
from datetime import timedelta
import pandas as pd
from scipy.io import wavfile
import re

chunk_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "chunks_3s"
wav_files = sorted(list(chunk_path.glob("*.wav")))
print(f"Found {len(wav_files)} .wav files in {chunk_path}")

if len(wav_files) > 1:
    example_file = wav_files[7]
    print(f"Example file: {example_file}")
else:
    raise ValueError("Not enough .wav files found in the directory.")

annotations_df = pd.read_csv(chunk_path.parents[1] / "annotations" / "annotations_clean.csv")

# A function is needed that takes wav file chunk and returns where in the recording it occurs
def chunk_to_time(file):
    # chunk is an integer from 000 to n where n is length of the original file, in seconds, divided by 3 (chunk size)
    # e.g. if file was 1 hour long, chunk can range from 000 to 1200 (3600 / 3)
    chunk_name = splitext(basename(file))[0]
    # all files named in format 'xxxx_chunk_yyy.wav'
    start_pos = chunk_name.find("k_") + 2
    chunk = chunk_name[start_pos:len(chunk_name)]
    if chunk == "000":
        secs = 0
    else:
        secs = int(chunk) * 3
    timestamp = str(timedelta(seconds=secs))
    return secs, timestamp


# It will be faster to find the corresponding label using seconds
def timestamp_in_secs(tstmp):
    h, m, s = map(int, tstmp.split(":"))
    return h*3600 + m*60 + s


def get_label_for_chunk(annotations, candidate_chunk_file, chunk_size=3):
    # Create two new columns containing the start and end points of all labels in seconds
    annotations["start_sec"] = annotations["start_time"].apply(timestamp_in_secs)
    annotations["end_sec"] = annotations["end_time"].apply(timestamp_in_secs)
    # Get seconds and timestamp of candidate chunk
    candidate_secs, candidate_ts = chunk_to_time(candidate_chunk_file)
    # Define start and end time of candidate chunk
    c_start = candidate_secs
    c_end = candidate_secs + (chunk_size - 1)  # Inclusive of all ms e.g. 000,000 to 002,999 not 003,000
    # Filter annotations accordingly
    # To capture chunks that fall within a labelled area and those that span > 1,
    # label start time must be <= candidate end time, and annotation end time must be >= candidate start time.
    labels_df = annotations[(annotations['start_sec'] <= c_end) & (annotations['end_sec'] >= c_start)]

    if labels_df.empty:
        return None
    else:
        return labels_df[['file', 'start_time', 'end_time', 'label']].to_dict(orient="records")


test = get_label_for_chunk(annotations_df, example_file)
print(test)
label_dir = chunk_path.parents[0] / "labelled_chunks_3s"

def qFunc(x):
    if re.search('/20240611_050000_', str(x)):
        return True
    else:
        return False

wav_filter = filter(qFunc, wav_files)
wav_subset = list(wav_filter)

for wav in wav_files:
    rate, data = wavfile.read(wav)
    # Define directory to save to
    os.makedirs(label_dir, exist_ok=True)
    # get labels, this returns list of dictionaries
    label_dict = get_label_for_chunk(annotations_df, wav)
    # returns just the labels in CHRONOLOGICAL ORDER
    labels = [d['label'] for d in label_dict]
    label_str = '_'.join(labels)  # join() is an instance method, needs instance.join i.e. 'some_string'.join()
    label_filename = f"{splitext(basename(wav))[0]}_{label_str}.wav"
    label_path = os.path.join(label_dir, label_filename)
    wavfile.write(label_path, rate, data)

