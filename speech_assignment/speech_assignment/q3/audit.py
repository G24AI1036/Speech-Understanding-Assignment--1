import librosa

audio, sr = librosa.load("../q1/audio.wav", sr=16000)

duration = len(audio) / sr

print("Audio Duration:", duration)

# Dummy bias example
print("Dataset Bias Report:")
print("Male: 60%")
print("Female: 40%")