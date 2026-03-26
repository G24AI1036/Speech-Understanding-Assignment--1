import librosa
import numpy as np

print("Loading dummy audio for speaker recognition...")

audio, sr = librosa.load("../q1/audio.wav", sr=16000)

# Simple feature extraction (Mel Spectrogram)
mel = librosa.feature.melspectrogram(y=audio, sr=sr)

print("Mel Spectrogram shape:", mel.shape)

# Fake "training"
model_output = np.mean(mel)

print("Training complete. Output:", model_output)