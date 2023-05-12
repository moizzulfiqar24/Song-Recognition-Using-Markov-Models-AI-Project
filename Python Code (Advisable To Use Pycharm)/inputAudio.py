import sounddevice as sd
import numpy as np
import pandas as pd
import librosa
import pickle
import tkinter as tk
from tkinter import ttk, font
import sys
import time

# ---------- Functions ------------------ #

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

def record_audio(duration, sample_rate=22050):
    print("\033[31mRecording audio...\033[0m")
    recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("\033[32mFinished recording\033[0m")
    return recorded_audio

def preprocess_audio(audio_data, sample_rate=22050):
    audio_data = audio_data / np.max(np.abs(audio_data))

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_data = librosa.resample(audio_data.flatten(), orig_sr=sample_rate, target_sr=sample_rate)

    return audio_data

def viterbi_algorithm(observed_features, model, n=1):
    viterbi_probs = []

    for i in range(len(observed_features) - n):
        state = tuple(observed_features[i:i + n])
        if state in model:
            next_states = model[state]
            match_probability = next_states.count(observed_features[i + n]) / len(next_states)
            viterbi_probs.append(match_probability)

    likelihood = sum(viterbi_probs)
    return likelihood

def recognize_song(mic_features_df, song_markov_models, n=1, threshold=0.1):
    mic_features = mic_features_df.iloc[0]

    features_list = []
    for feature_name in ["mfccs", "spectral_centroid", "tempo", "spectral_contrast", "spectral_rolloff"]:
        features_list.extend(mic_features[feature_name])

    max_likelihood = -1
    best_match = None

    for song_model in song_markov_models:
        likelihood = viterbi_algorithm(features_list, song_model["model"], n)

        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_match = song_model

    if max_likelihood >= threshold:
        return best_match["title"], best_match["artist"]
    else:
        return None, None

# ---------- UI Functions ------------------ #

def start_recording_and_recognize_song():
    global recorded_audio
    global song_markov_models

    duration = 30
    sample_rate = 22050
    recorded_audio = record_audio(duration, sample_rate)

    preprocessed_audio = preprocess_audio(recorded_audio)
    audio_features = extract_audio_features(preprocessed_audio, sample_rate)  

    mic_input_data = {
        "title": "Microphone Input",
        "artist": "User",
    }
    mic_input_data.update(audio_features)
    mic_features_df = pd.DataFrame([mic_input_data])

    # Save the DataFrame as a CSV file
    mic_features_df.to_csv("mic_features.csv", index=False)

    # Read the CSV file
    mic_features_df = pd.read_csv("mic_features.csv")

    n = 3 
    recognized_title, recognized_artist = recognize_song(mic_features_df, song_markov_models, n)

    if recognized_title and recognized_artist:
        result_text = f"The Song Name Is \"{recognized_title}\" by {recognized_artist}"
    else:
        result_text = "The song cannot be found in the database."

    result_label.config(text=result_text)
    print(result_text)
    # time.sleep(10)
    # sys.exit()

# ---------- Load Song Markov Models ------------------ #

with open("song_markov_models.pkl", "rb") as f:
    song_markov_models = pickle.load(f)

# ---------- UI ------------------ #

root = tk.Tk()
root.title("Song Recognizer")
root.geometry("1200x450")
root.configure(bg="#35374C")

# Create a custom font
custom_font = font.nametofont("TkDefaultFont")
custom_font.configure(size=16)

style = ttk.Style()
style.configure("TButton", font=custom_font, background="#ffffff", foreground="#c23225")
style.configure("TLabel", font=custom_font, background="#ffffff", foreground="#c23225")

frame = ttk.Frame(root, padding=(30, 30, 30, 30))
frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(frame, text="Song Recognizer", font=("Comic Sans MS", 28, "bold"), background="#ffffff", foreground="#35374C")
title_label.pack(pady=(0, 20))

record_button = ttk.Button(frame, text="Start Recording", command=start_recording_and_recognize_song)
record_button.pack(fill=tk.BOTH, pady=(0, 20))

result_label = ttk.Label(frame, text="", wraplength=300, justify="center", font=("Comic Sans MS", 16, "bold"), background="#ffffff", foreground="#111691")
result_label.pack()

root.mainloop()
