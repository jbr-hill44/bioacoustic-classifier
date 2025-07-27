import pandas as pd

annotations_df = pd.read_csv("data/annotations/annotations.csv")
annotations_copy = annotations_df.copy()

# Since labelling, now know that unknown_bird_1 = yellowhammer and unknown_bird_6 = wren
annotations_copy.loc[annotations_copy['label'] == 'unknown_bird_1', 'label'] = 'yellowhammer'
annotations_copy.loc[annotations_copy['label'] == 'unknown_bird_6', 'label'] = 'wren'

# Records labelled with 'hooded_crow' should actually be 'carrion_crow'
# Labels containing this need replacing
annotations_copy['label'] = annotations_copy['label'].replace('hooded_crow', 'carrion_crow', regex=True)

# One of the labels that is two rows should just be one.
# Drop row by index, then replace value in next row with start time of removed row (00:43:28)
annotations_copy = annotations_copy.drop([298], axis=0)
annotations_copy.loc[annotations_copy['start_time'] == '00:43:31', 'start_time'] = '00:43:28'
annotations_copy = annotations_copy.reset_index()
annotations_copy = annotations_copy.drop('index', axis=1)

# Write back to original (scary)
annotations_df = annotations_copy

# Write new csv
annotations_df.to_csv('data/annotations/annotations_clean.csv', index=False)