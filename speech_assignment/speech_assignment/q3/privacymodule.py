import librosa

def transform_voice(audio, sr):
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
    return shifted