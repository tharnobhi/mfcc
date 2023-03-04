import os
import pandas as pd
import librosa
import numpy as np
from datetime import datetime

# Set the path to the directory containing the WAV files
audio_path = "audio data/vad/"

# Define a function to extract MFCCs from an audio file
def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs_all = np.vstack((mfccs, mfccs_delta, mfccs_delta2))
    return mfccs_all.T

# Initialize empty lists to store the MFCCs, labels, and timestamps
mfccs_data = []
labels = []
timestamps = []

# Loop through all the audio files in the directory and extract their MFCCs
for filename in os.listdir(audio_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_path, filename)
        label = filename.split("_")[0]  # extract the label from the filename
        mfccs = extract_mfcc(file_path)
        mfccs_avg = np.mean(mfccs, axis=0)  # compute the average MFCCs across time
        mfccs_data.append(mfccs_avg)
        labels.append(label)
        timestamp = os.path.getmtime(file_path)
        timestamps.append(datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))

# Convert the MFCCs, labels, and timestamps to data frames using pandas
mfccs_df = pd.DataFrame(mfccs_data, columns=["mfcc_" + str(i) for i in range(39)])
labels_df = pd.DataFrame(labels, columns=["label"])
timestamps_df = pd.DataFrame(timestamps, columns=["timestamp"])

# Concatenate the MFCCs, labels, and timestamps into a single data frame
data = pd.concat([mfccs_df, labels_df, timestamps_df], axis=1)

# Save the data frame to a CSV file
data.to_csv("mfcc_data.csv", index=False)
