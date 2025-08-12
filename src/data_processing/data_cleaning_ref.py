import pandas as pd

annotations_df = pd.read_csv("data/annotations/annotations.csv")
annotations_copy = annotations_df.copy()

# First row is erroneous repeat
annotations_copy = annotations_copy.drop([0], axis=0)
annotations_copy = annotations_copy.reset_index()
annotations_copy = annotations_copy.drop('index', axis=1)

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
    'unknown_bird_7': 'pheasant'
})

annotations_copy['label'] = annotations_copy['label'].replace({
    r'unknown_bird_1(_and.*)?$': 'yellowhammer',
    r'hooded_crow(_and.*)?$': 'carrion_crow',
    r'chirping_1?$': 'chirping'
},
regex=True)


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

error_count = []
for index in range(len(annotations_copy)-1):
    endsec = annotations_copy.iloc[index, 5]
    startsec = annotations_copy.iloc[(index+1), 4]
    if not startsec == (endsec + 1):
        error_count.append(annotations_copy.iloc[index, :])


# Write back to original (scary)
annotations_df = annotations_copy

# Write new csv
annotations_df.to_csv('data/annotations/annotations_clean.csv', index=False)