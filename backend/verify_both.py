from audio_processor import AudioProcessor
from model_handler import VoiceDetector
import numpy as np

proc = AudioProcessor(sr=16000)
det = VoiceDetector()

sr = 16000
duration = 2.5
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 1. Human Voice test
f0_h = 135 + 14 * np.sin(2 * np.pi * 1.8 * t)
phase_h = 2 * np.pi * np.cumsum(f0_h / sr)
y_human = 0.4 * np.sin(phase_h) + 0.6 * np.sin(2 * phase_h)
y_human *= (0.5 + 0.3 * np.sin(2 * np.pi * 2 * t))
y_human += np.random.randn(len(y_human)) * 0.005
y_human = (y_human / np.max(np.abs(y_human)) * 0.5).astype(np.float32)

res_h = det.predict(y_human, sr)
print(f"Human Voice: Score = {res_h['score']} -> {res_h['label']}")
for r in res_h['reasons']:
    print(f"  {r}")

# 2. AI Voice test (Gemini / ChatGPT)
f0_ai = 160 + 55 * np.sin(2 * np.pi * 5 * t) + 30 * np.sin(2 * np.pi * 11 * t)
phase_ai = 2 * np.pi * np.cumsum(f0_ai / sr)
y_ai = 0.5 * np.sin(phase_ai) + 0.4 * np.sin(2 * phase_ai)
y_ai += np.random.randn(len(y_ai)) * 0.04
y_ai = (y_ai / np.max(np.abs(y_ai)) * 0.5).astype(np.float32)

res_ai = det.predict(y_ai, sr)
print(f"\nGemini / AI Voice: Score = {res_ai['score']} -> {res_ai['label']}")
for r in res_ai['reasons']:
    print(f"  {r}")
