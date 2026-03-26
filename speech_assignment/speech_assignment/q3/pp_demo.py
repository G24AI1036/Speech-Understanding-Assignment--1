import librosa
from privacymodule import transform_voice

audio, sr = librosa.load("../q1/audio.wav", sr=16000)
new_audio = transform_voice(audio, sr)

print("Privacy transformation applied")