import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_FILE = ANNOTATION_DIR / "annotations.csv"

if ANNOTATION_FILE.exists():
    annotations_df = pd.read_csv(ANNOTATION_FILE)
else:
    annotations_df = pd.DataFrame(columns=["file","start_time","end_time","label"])

st.title("Data Annotation App")

raw_files = sorted((PROJECT_ROOT / "data" / "raw").glob("*.WAV"))
file_names = [f.name for f in raw_files]

selected_file = st.selectbox("Select file to label", file_names)

start_time = st.text_input("Enter label start time in format HH:MM:SS")
end_time = st.text_input("Enter label end time in format HH:MM:SS")
label = st.text_input("Enter label")

if st.button("Add label"):
    if selected_file and start_time and end_time and label:
        new_row = {"file": selected_file, "start_time": start_time, "end_time": end_time, "label": label}
        annotations_df = pd.concat([annotations_df,pd.DataFrame([new_row])], ignore_index=True)
        annotations_df.to_csv(ANNOTATION_FILE, index=False)
        st.success(f"Added label: {label} for {selected_file} ({start_time}-{end_time})")
    else:
        st.error("Please complete all fields")

st.subheader("Existing Annotations")
# Should alter this so that it filters the dataframe to only show those labels relating to the selected file
st.dataframe(annotations_df)