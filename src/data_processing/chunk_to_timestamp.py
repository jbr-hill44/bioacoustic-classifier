from pathlib import Path
import os
from os.path import splitext, basename
from datetime import timedelta
import pandas as pd
from scipy.io import wavfile
import re

# Define what we need
# Location of 3s chunks
chunk_path = Path().resolve().parents[1] / "data" / "processed" / "chunks_3s"
#chunk_path = Path.cwd() / "data" / "processed" / "chunks_3s"
# get files into lis
wav_files = sorted(list(chunk_path.glob("*.wav")))
# annotations taken from recording
annotations_df = pd.read_csv(chunk_path.parents[1] / "annotations" / "annotations_clean.csv")
# define a directory for the labels
label_dir = chunk_path.parents[0] / "labelled_chunks_3s"
# filter function for only those chunks that have been labelled
def qFunc(x):
    if re.search('/20240611_050000_', str(x)):
        return True
    elif re.search('/20240612_050000_', str(x)):
        return True
    else:
        return False
# apply filter
wav_filter = filter(qFunc, wav_files)
wav_subset = list(wav_filter)

# Further cleaning is required due to how labels are combined
# First, we need to create a list/dataframe of all single labels
# This can include: wren, eurasian_skylark, unknown_bird_12 <- all have 2 or fewer underscores
one_label_df = annotations_df[annotations_df['label'].str.count('_') <= 2]
# From this, exclude joined single labels like wren_and_rumble
one_label_df = one_label_df[one_label_df['label'].str.count('and') == 0]
# Edge case: if all instances of a string containing no underscore are doubled.
# E.g. if wren only appears as wren_wren and never just wren
one_label = list(one_label_df['label'].unique())
one_label.append('unknown_bird_2')

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
    # Get seconds and timestamp of candidate chunk
    candidate_secs, candidate_ts = chunk_to_time(candidate_chunk_file)
    name_root = f'{os.path.splitext(os.path.basename(candidate_chunk_file))[0].split('_chunk_')[0]}.WAV'
    # Define start and end time of candidate chunk
    c_start = candidate_secs
    c_end = candidate_secs + (chunk_size - 1)  # Inclusive of all ms e.g. 000,000 to 002,999 not 003,000
    # Filter annotations accordingly
    # To capture chunks that fall within a labelled area and those that span > 1,
    # label start time must be <= candidate end time, and annotation end time must be >= candidate start time.
    labels_df = annotations[(annotations['file'] == name_root) & (annotations['start_sec'] <= c_end) & (annotations['end_sec'] >= c_start)]

    if labels_df.empty:
        return None
    else:
        return labels_df[['file', 'start_time', 'end_time', 'label']].to_dict(orient="records")

def clean_strings(labels, label_filename):
    special_case = re.compile(r'^unknown_bird_\d+_and_unknown_bird_\d+$')
    # remove repeats
    for lab in labels:
        if special_case.fullmatch(label_filename):
            continue
        if len(re.findall(rf'({lab})', label_filename)) > 1:
            while len(re.findall(rf'({lab})', label_filename)) > 1:
                label_filename = re.sub(rf'(_and_{lab})', '', string=label_filename, count=1)
    # normalise
    match = re.search(r'([0-9]{8}_[0-9]{6}_chunk_[0-9]+_)(.*?)(_and_)(.*?)(.wav)$', label_filename)
    if match:
        part1 = match.group(1)
        part2 = match.group(2)
        part3 = match.group(3)
        part4 = match.group(4)
        part5 = match.group(5)

        # Sort the two parts alphabetically
        sorted_parts = sorted([part2, part4])
        label_filename = f"{part1}{sorted_parts[0]}{part3}{sorted_parts[1]}{part5}"
    return label_filename

def label_wavs(wav_file, label_dir, annotations):
    rate, data = wavfile.read(wav_file)
    # Define directory to save to
    os.makedirs(label_dir, exist_ok=True)
    # get labels, this returns list of dictionaries
    label_dict = get_label_for_chunk(annotations, wav_file)
    if not label_dict:
        return
    # returns just the labels in CHRONOLOGICAL ORDER
    labels = [d['label'] for d in label_dict]
    label_str = '_and_'.join(labels)  # join() is an instance method, needs instance.join i.e. 'some_string'.join()
    label_filename = f"{splitext(basename(wav_file))[0]}_{label_str}.wav"
    label_filename = clean_strings(one_label, label_filename)
    label_path = os.path.join(label_dir, label_filename)
    wavfile.write(label_path, rate, data)
    print(f'labelled {label_path}')


for wav in wav_subset:
    label_wavs(wav_file=wav, label_dir=label_dir, annotations=annotations_df)

# Quick and dirty fixes because I am done with regex for now
old_path = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_271_eurasian_skylark_2_and_unknown_bird_2.wav')
new_path = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_271_eurasian_skylark_and_unknown_bird_2.wav')
old_path.rename(new_path)
old_path1 = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_865_unknown_bird_11_12.wav')
old_path2 = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_867_unknown_bird_12_13.wav')
new_path1 = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_865_unknown_bird_11_and_unknown_bird_12.wav')
new_path2 = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_867_unknown_bird_11_and_unknown_bird_12.wav')
old_path1.rename(new_path1)
old_path2.rename(new_path2)
old_path3 = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_892_unknown_bird_15_13.wav')
new_path3 = Path('/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/labelled_chunks_3s/20240611_050000_chunk_892_unknown_bird_15_and_unknown_bird_13.wav')
old_path3.rename(new_path3)
