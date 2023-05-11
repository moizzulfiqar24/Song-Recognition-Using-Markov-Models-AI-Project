import os
import librosa
import pandas as pd
import pickle

# ---------- Functions ------------------ ##

def extract_audio_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    avg_mfccs = mfccs.mean(axis=1)
    avg_spectral_centroid = spectral_centroid.mean(axis=1)
    avg_spectral_contrast = spectral_contrast.mean(axis=1)
    avg_spectral_rolloff = spectral_rolloff.mean(axis=1)

    return {
        "mfccs": list(avg_mfccs),
        "spectral_centroid": list(avg_spectral_centroid),
        "tempo": [tempo],
        "spectral_contrast": list(avg_spectral_contrast),
        "spectral_rolloff": list(avg_spectral_rolloff)
    }

# ---------- Step 1 ----------------- #

audio_files_path = "audio_files/"
audio_files = os.listdir(audio_files_path)

song_features_list = []

for file in audio_files:
    print(f"\033[32mCurrently Processing Song: \033[0m{file}") # debugging
    file_path = os.path.join(audio_files_path, file)
    y, sr = librosa.load(file_path)
    features = extract_audio_features(y, sr)

    # As file names are in the format 'artist - song_title.mp3'
    file_name = os.path.splitext(file)[0]
    split_file_name = file_name.split(' - ')

    if len(split_file_name) == 2:
        artist, title = split_file_name
    else:
        artist = "Unknown"
        title = file_name

    song_data = {
        "title": title,
        "artist": artist,
    }
    song_data.update(features)
    song_features_list.append(song_data)

# Create a DataFrame from the list of dictionaries
song_features_df = pd.DataFrame(song_features_list)

print("\033[32m\nSong Features Table :-\033[0m")
print(song_features_df)

# Save the DataFrame as a CSV file
song_features_df.to_csv("song_features.csv", index=False)

# ---------- Step 1 Complete ----------------- #

song_features_df = pd.read_csv("song_features.csv")
def build_markov_model(features, n=1):
    model = {}
    for i in range(len(features) - n):
        state = tuple(features[i:i + n])
        next_state = features[i + n]

        if state not in model:
            model[state] = []
        model[state].append(next_state)
    return model

def build_models_for_all_songs(df, n=1):
    markov_models = []
    for _, row in df.iterrows():
        song_data = {
            "title": row["title"],
            "artist": row["artist"],
        }

        features_list = []
        for feature_name in ["mfccs", "spectral_centroid", "tempo", "spectral_contrast", "spectral_rolloff"]:
            features_list.extend(row[feature_name])

        model = build_markov_model(features_list, n)
        song_data["model"] = model
        markov_models.append(song_data)
    return markov_models

# Build Markov models for all songs
n = 3
song_markov_models = build_models_for_all_songs(song_features_df, n)

# Save the Markov models
with open("song_markov_models.pkl", "wb") as f:
    pickle.dump(song_markov_models, f)