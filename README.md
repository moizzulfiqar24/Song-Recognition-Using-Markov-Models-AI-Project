# Song Recognizer

This project contains two Python scripts, `songsProcessing.py` and `inputAudio.py`, that work together to recognize songs based on their audio features. The first script, `songsProcessing.py`, processes a set of audio files, extracts their features, and builds a Markov model for each song. The second script, `inputAudio.py`, records audio from the user's microphone, processes it, and then attempts to recognize the song based on the previously built Markov models.

## Dependencies

To run the scripts, you'll need the following Python libraries:

- `os`
- `librosa`
- `pandas`
- `pickle`
- `sounddevice`
- `numpy`
- `tkinter`

You can install them using pip:

```
pip install librosa pandas numpy sounddevice tkinter
```

## Usage

1. Place your audio files (MP3 format) in a folder called `audio_files` in the same directory as the `songsProcessing.py` script.
2. Run the `songsProcessing.py` script:

```
python songsProcessing.py
```

This script will extract audio features from each audio file and save the results to a CSV file (`song_features.csv`) and a pickle file containing the Markov models (`song_markov_models.pkl`).

3. Run the `inputAudio.py` script:

```
python inputAudio.py
```

This script will launch a graphical user interface (GUI) that allows you to record audio from your microphone. Click the "Start Recording" button and record a 30-second clip of the song you want to recognize. The script will then attempt to identify the song based on its audio features and the previously built Markov models. The result will be displayed in the GUI.

## Files

- `songsProcessing.py`: Script to process audio files and build Markov models.
- `inputAudio.py`: Script to record audio, process it, and attempt to recognize the song using the Markov models.
- `audio_files/`: Directory containing audio files (MP3 format) to be processed.
- `song_features.csv`: CSV file containing extracted audio features for each song.
- `song_markov_models.pkl`: Pickle file containing Markov models for each song.
- `mic_features.csv`: CSV file containing extracted audio features from the recorded microphone input.

## Note

- The default `n` value (number of previous states in the Markov model) is set to 3. You can experiment with this value to improve the model's performance.
- The default `threshold` value (minimum likelihood for a song to be considered a match) is set to 0.1. You can experiment with this value to adjust the model's sensitivity.
