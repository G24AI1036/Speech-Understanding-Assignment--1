import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

audio, sr = librosa.load("audio.wav", sr=16000)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(**inputs).logits

pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)
print("Using Wav2Vec2 base model for transcription")
print("Transcription:", transcription[0])