import pandas as pd
import numpy as np

annotations_df = pd.read_csv("data/annotations/annotations.csv")
annotations_copy = annotations_df.copy()

# First row is erroneous repeat
annotations_copy = annotations_copy.drop([0], axis=0)
annotations_copy = annotations_copy.reset_index()
annotations_copy = annotations_copy.drop('index', axis=1)

# One of the labels that is two rows should just be one.
# Drop row by index, then replace value in next row with start time of removed row (00:43:28)
annotations_copy = annotations_copy.drop([298], axis=0)  # identified through manual inspection
annotations_copy.loc[annotations_copy['start_time'] == '00:43:31', 'start_time'] = '00:43:28'
annotations_copy = annotations_copy.reset_index()
annotations_copy = annotations_copy.drop('index', axis=1)
# Repeat for rows where end and start time are non-contiguous
annotations_copy = annotations_copy.drop([30], axis=0)  # identified through manual inspection
annotations_copy.loc[annotations_copy['start_time'] == '00:10:35', 'start_time'] = '00:10:36'
annotations_copy = annotations_copy.reset_index()
annotations_copy = annotations_copy.drop('index', axis=1)
annotations_copy.loc[annotations_copy['end_time'] == '00:18:20', 'end_time'] = '00:18:21'
annotations_copy.loc[annotations_copy['start_time'] == '00:18:23', 'start_time'] = '00:18:22'
annotations_copy.loc[annotations_copy['start_time'] == '00:38:45', 'start_time'] = '00:38:44'
annotations_copy.loc[annotations_copy['start_time'] == '00:40:36', 'start_time'] = '00:40:32'

# Need to ensure we have contiguous labels (no time unaccounted for)
annotations_copy['start_sec'] = annotations_copy['start_time'].apply(timestamp_in_secs)
annotations_copy['end_sec'] = annotations_copy['end_time'].apply(timestamp_in_secs)

new_annotations = pd.read_csv('data/annotations/raw annotations/20240612_050000.txt', sep ='\t')
new_annotations['file'] = '20240612_050000.WAV'
new_annotations['start_sec'] = np.floor(new_annotations['Begin Time (s)']).astype(int)
new_annotations['end_sec'] = np.floor(new_annotations['End Time (s)']).astype(int)
new_annotations = new_annotations.rename(columns={
'Annotation': 'label'})
new_annotations['start_time'] = new_annotations['start_sec'].apply(
    lambda x: f"{x//3600:02}:{(x%3600)//60:02}:{x%60:02}"
)
new_annotations['end_time'] = new_annotations['end_sec'].apply(
    lambda x: f"{x//3600:02}:{(x%3600)//60:02}:{x%60:02}"
)
new_annotations['end_sec'] = np.floor(new_annotations['End Time (s)']).astype(int)
new_annotations = new_annotations.loc[:,
                  ['file', 'label','start_time', 'end_time','start_sec','end_sec']]

annotations_copy = pd.concat([annotations_copy, new_annotations], axis=0, ignore_index=True)

# Since labelling, now know that unknown_bird_1 = yellowhammer and unknown_bird_6 = wren
# Records labelled with 'hooded_crow' should actually be 'carrion_crow'
# also correct spelling mistakes
# Need to remove adjectives (e.g. quiet)
annotations_copy['label'] = annotations_copy['label'].replace({
    'quiet_eurasian': 'eurasian',
    'unkown': 'unknown'},
    regex=True
)
annotations_copy['label'] = annotations_copy['label'].replace({
    'hooded_crow': 'carrion_crow',
    'yellow_hammer': 'yellowhammer',
    'unknown_bird_1': 'yellowhammer',
    'unknown_bird_6': 'wren',
    'unknown_bird_8': 'wren',
    'background': 'background_noise',
    'unknown_bird_7': 'pheasant',
    'woodpiegon': 'woodpigeon',
    'yellowhammer_blackcap': 'yellowhammer_and_blackcap',
    'metallic_bang': 'background_noise',
    'rustling': 'background_noise',
    'unknown_chirping_and_metallic_bang': 'background_noise',
    'rumble': 'background_noise',
    'dog_barking': 'background_noise'
})

annotations_copy['label'] = annotations_copy['label'].replace({
    r'unknown_bird_1(_and.*)?$': 'yellowhammer',
    r'hooded_crow(_and.*)?$': 'carrion_crow',
    r'chirping_1?$': 'chirping',
    r'metallic_bang$': 'background_noise',
    r'rumble$': 'background_noise',
    r'metal$': 'background_noise',
    r'dog_barking$': 'background_noise'
},
regex=True)

annotations_copy['label'] = annotations_copy['label'].replace({
    r'unknown_chirping$': 'background_noise'},
regex=True)

annotations_copy['label'] = annotations_copy['label'].replace({
    'background_noise_and_background_noise': 'background_noise'})

annotations_copy = annotations_copy.drop(annotations_copy.index[514:520], axis=0)

error_count = []
for index in range(len(annotations_copy)-1):
    endsec = annotations_copy.iloc[index, 5]
    startsec = annotations_copy.iloc[(index+1), 4]
    if not ((startsec == (endsec + 1)) | (startsec == endsec)):
        error_count.append(annotations_copy.iloc[index, :])


# Write back to original (scary)
annotations_df = annotations_copy

# Write new csv
annotations_df.to_csv('data/annotations/annotations_clean.csv', index=False)