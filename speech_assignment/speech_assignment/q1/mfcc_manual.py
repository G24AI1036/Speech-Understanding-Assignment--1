import numpy as np
import librosa
from scipy.fftpack import dct
import matplotlib.pyplot as plt

audio_path = "audio.wav"
signal, sr = librosa.load(audio_path, sr=16000)

# Pre-emphasis
pre_emphasis = 0.99
emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# Framing
frame_size = 0.034
frame_stride = 0.02

frame_length = int(frame_size * sr)
frame_step = int(frame_stride * sr)

num_frames = int(np.ceil(float(np.abs(len(emphasized) - frame_length)) / frame_step))

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - len(emphasized)))
pad_signal = np.append(emphasized, z)

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
    np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
).T

frames = pad_signal[indices.astype(np.int32, copy=False)]

# Windowing
frames *= np.hamming(frame_length)

# FFT
NFFT = 256
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)

# Mel filter banks
nfilt = 34
low_mel = 0
high_mel = 2595 * np.log10(1 + (sr / 2) / 700)

mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
hz_points = 700 * (10 ** (mel_points / 2595) - 1)

bins = np.floor((NFFT + 1) * hz_points / sr)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bins[m - 1])
    f_m = int(bins[m])
    f_m_plus = int(bins[m + 1])

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

log_energy = np.log(filter_banks)

mfcc = dct(log_energy, type=2, axis=1, norm="ortho")[:, :13]

print("MFCC shape:", mfcc.shape)

plt.imshow(mfcc.T, aspect="auto", origin="lower")
plt.title("MFCC Features")
plt.show()