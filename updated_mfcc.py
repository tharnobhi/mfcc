import os
import pandas as pd
import librosa
import numpy as np

# Set the path to the directory containing the WAV files
audio_path = "test audio/"

# Define a function to extract MFCCs from an audio file
def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=16000) # change the sampling rate to 16 kHz
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=int(sr*0.03), hop_length=int(sr*0.01), n_mels=30) # modify MFCC parameters
    return mfccs.T

# Initialize empty list to store the MFCCs
mfccs_data = []

# Loop through all the audio files in the directory and extract their MFCCs
for filename in os.listdir(audio_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_path, filename)
        mfccs = extract_mfcc(file_path)
        mfccs_avg = np.mean(mfccs, axis=0)  # compute the average MFCCs across time
        mfccs_data.append(mfccs_avg)

# Convert the MFCCs to a data frame using pandas
mfccs_df = pd.DataFrame(mfccs_data, columns=["mfcc_" + str(i) for i in range(20)])

# Save the data frame to a CSV file
mfccs_df.to_csv("mfcc_data.csv", index=False)
