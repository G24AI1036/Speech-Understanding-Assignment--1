import numpy as np
import librosa
import matplotlib.pyplot as plt

signal, sr = librosa.load("audio.wav", sr=16000)

frame = signal[:2048]

windows = {
    "Hamming": np.hamming(len(frame)),
    "Rectangular": np.ones(len(frame)),
    "Hanning": np.hanning(len(frame)),
}
for name, w in windows.items():
    windowed = frame * w
    spectrum = np.abs(np.fft.fft(windowed))

    signal_power = np.mean(windowed ** 2)
    noise_power = np.var(windowed)

    snr = 10 * np.log10(signal_power / noise_power)
    print(name, "SNR:", snr)

    plt.plot(spectrum[:1000], label=name)

plt.legend()
plt.title("Spectral Leakage Comparison")
plt.show()