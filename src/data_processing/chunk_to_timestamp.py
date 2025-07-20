import os.path
from pathlib import Path
from os.path import splitext
from datetime import timedelta

chunk_path = Path(__file__).resolve() / "data" / "processed" / "chunks_3s"

# A function is needed that takes wav file chunk and returns where in the recording it occurs
def chunk_to_time(file):
    # chunk is an integer from 000 to n where n is length of the original file, in seconds, divided by 3 (chunk size)
    # e.g. if file was 1 hour long, chunk can range from 000 to 1200 (3600 / 3)
    chunk_name = os.path.splitext(os.path.basename(file))[0]
    start_pos = chunk_name.find("k_") + 2
    chunk = chunk_name[start_pos:len(chunk_name)]
    if(chunk == "000"):
        secs = 0
    else:
        secs = int(chunk) * 3
    timestamp = str(timedelta(seconds=secs))
    return timestamp