import librosa
import numpy as np
import matplotlib.pyplot as plt

signal, sr = librosa.load("audio.wav", sr=16000)

frame_length = 400
hop = 160

frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop).T

energy = np.sum(frames ** 2, axis=1)

threshold = np.mean(energy) * 1.1
voiced = energy > threshold

plt.plot(energy)
plt.axhline(threshold, color="red")
plt.title("Voiced / Unvoiced Detection")
plt.show()

print("Voiced frames:", np.sum(voiced))
print("Unvoiced frames:", len(voiced) - np.sum(voiced))