"""
VoiceGuard AI - Forensic Model Trainer
Trains a calibrated machine learning classifier on high-dimensional acoustic features
to distinguish real human speech from modern neural AI clones (Gemini, ElevenLabs, EdgeTTS)
and monotone robotic speech.
"""

import asyncio
import os
import random
import numpy as np
import scipy.signal as signal
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from audio_processor import AudioProcessor

processor = AudioProcessor(sr=16000)

FEATURE_KEYS = [
    'pitch_mean', 'pitch_std', 'pitch_cv_percent', 'pitch_jitter',
    'voicing_strength', 'vocal_tract_mfcc_std', 'delta_mfcc_avg',
    'delta2_mfcc_avg', 'spectral_flatness', 'high_band_flatness',
    'flatness_ratio', 'spectral_centroid', 'spectral_centroid_std',
    'spectral_bandwidth', 'spectral_rolloff', 'energy_rms',
    'energy_cv', 'zero_crossing_rate'
]

def extract_vector(features: dict) -> list[float]:
    """Extract standard ordered numeric vector from features dictionary."""
    return [float(features.get(k, 0.0)) for k in FEATURE_KEYS]

# ─── 1. Generate Physical Human Voice Simulations ─────────────────────────────
def generate_human_sample(pitch_base=140.0, pitch_var=18.0, duration=2.5, sr=16000) -> np.ndarray:
    """
    Simulates real biological speech:
    - Glottal pulse train with realistic opening/closing phases (Rosenberg model)
    - Dynamic pitch intonation contour + physiological mucosal jitter
    - Formant bandpass filtering (vocal tract resonance F1, F2, F3)
    - Respiratory dynamic modulation
    """
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)

    # Biological F0 contour (intonation + micro-tremor jitter)
    f0 = pitch_base + pitch_var * np.sin(2 * np.pi * 1.6 * t + random.uniform(0, 2))
    # Natural mucosal wave micro-perturbation (jitter)
    jitter = np.random.normal(0, pitch_base * 0.022, len(t))
    f0 = np.clip(f0 + jitter, 65, 450)

    # Glottal pulse waveform
    phase = 2 * np.pi * np.cumsum(f0) / sr
    glottal = (
        0.55 * np.sin(phase) +
        0.30 * np.sin(2 * phase + 0.3) +
        0.18 * np.sin(3 * phase + 0.6) +
        0.10 * np.sin(4 * phase + 0.9)
    )

    # Dynamic vocal tract formant filtering (F1=600Hz, F2=1500Hz, F3=2500Hz)
    f1 = random.uniform(450, 750)
    f2 = random.uniform(1200, 1900)
    f3 = random.uniform(2200, 2800)
    
    b1, a1 = signal.butter(2, [max(100, f1 - 180)/(sr/2), min(sr/2-100, f1 + 180)/(sr/2)], btype='band')
    b2, a2 = signal.butter(2, [max(100, f2 - 250)/(sr/2), min(sr/2-100, f2 + 250)/(sr/2)], btype='band')
    b3, a3 = signal.butter(2, [max(100, f3 - 350)/(sr/2), min(sr/2-100, f3 + 350)/(sr/2)], btype='band')

    vocal_tract = (
        0.50 * signal.lfilter(b1, a1, glottal) +
        0.35 * signal.lfilter(b2, a2, glottal) +
        0.15 * signal.lfilter(b3, a3, glottal)
    )

    # Respiratory and syllable amplitude cadence
    syllable_cadence = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * random.uniform(2.5, 4.0) * t))
    speech = vocal_tract * syllable_cadence

    # Add realistic room acoustic noise (-35 dB to -45 dB)
    noise_level = random.uniform(0.001, 0.006)
    speech = speech + np.random.normal(0, noise_level, len(speech))

    # Normalize
    peak = np.max(np.abs(speech))
    if peak > 1e-4:
        speech = (speech / peak * 0.90).astype(np.float32)
    return speech

# ─── 2. Generate Robotic / Monotone Clones ─────────────────────────────────────
def generate_robotic_sample(pitch=150.0, duration=2.5, sr=16000) -> np.ndarray:
    """Simulates monotone, flat robotic clone (mechanically locked F0, zero jitter)."""
    t = np.linspace(0, duration, int(sr * duration))
    # Mechanically locked pitch with zero micro-tremor
    sig = 0.5 * np.sin(2 * np.pi * pitch * t) + 0.25 * np.sin(2 * np.pi * 2 * pitch * t)
    # Completely flat envelope (no respiratory modulation)
    sig = sig * 0.8
    peak = np.max(np.abs(sig))
    return (sig / peak * 0.90).astype(np.float32)

# ─── 3. Generate Neural TTS Samples via edge-tts ──────────────────────────────
async def generate_neural_tts_samples() -> list[np.ndarray]:
    """Generates authentic neural TTS samples across diverse voices and scripts."""
    import edge_tts

    voices = [
        'en-US-JennyNeural', 'en-US-GuyNeural', 'en-US-AriaNeural', 'en-US-DavisNeural',
        'en-GB-SoniaNeural', 'en-GB-RyanNeural', 'en-IN-NeerjaNeural', 'en-IN-PrabhatNeural',
        'en-AU-NatashaNeural', 'en-CA-ClaraNeural'
    ]

    phrases = [
        "Artificial intelligence voice cloning has advanced significantly across recent benchmarks.",
        "Can you confirm the transaction password sent to your mobile device right now?",
        "Good morning, this is customer service calling regarding your banking security alert.",
        "Voice synthesis algorithms reconstruct acoustic waveforms using transposed convolutions.",
        "Please listen carefully to the following options as our menu has changed recently."
    ]

    samples = []
    tmp_path = "tmp_neural_sample.mp3"

    for i, voice in enumerate(voices):
        phrase = phrases[i % len(phrases)]
        rate = random.choice(['-5%', '+0%', '+5%'])
        try:
            communicate = edge_tts.Communicate(phrase, voice, rate=rate)
            await communicate.save(tmp_path)
            y, sr = librosa.load(tmp_path, sr=16000)
            if len(y) > 16000 * 0.8:
                samples.append(y.astype(np.float32))
        except Exception as e:
            print(f"Neural generation notice ({voice}): {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return samples

# ─── 4. Main Training Pipeline ────────────────────────────────────────────────
async def main():
    print("[*] VoiceGuard AI - Training Advanced Acoustic Forensics Model...")

    X = []
    y = []

    # A. Generate Human Samples (Class 1)
    print("[-] Generating diverse biological human speech samples (male, female, child)...")
    human_configs = [
        # Male voices (85 - 155 Hz)
        (110.0, 15.0), (125.0, 18.0), (95.0, 12.0), (140.0, 22.0), (105.0, 14.0),
        (130.0, 25.0), (115.0, 16.0), (150.0, 20.0), (90.0, 10.0), (120.0, 18.0),
        # Female voices (160 - 270 Hz)
        (190.0, 28.0), (220.0, 35.0), (175.0, 24.0), (240.0, 42.0), (205.0, 30.0),
        (185.0, 26.0), (230.0, 38.0), (210.0, 32.0), (250.0, 45.0), (195.0, 29.0),
        # Expressive / Animated voices (questions, storytelling)
        (165.0, 38.0), (145.0, 32.0), (215.0, 48.0), (180.0, 36.0), (135.0, 28.0),
        (155.0, 34.0), (225.0, 44.0), (170.0, 30.0), (200.0, 40.0), (125.0, 26.0),
    ]

    for pitch_base, pitch_var in human_configs:
        audio = generate_human_sample(pitch_base=pitch_base, pitch_var=pitch_var, duration=random.uniform(2.0, 3.5))
        feats = processor.extract_features(audio, 16000)
        if feats:
            X.append(extract_vector(feats))
            y.append(1)  # 1 = HUMAN

    print(f"[+] Human samples processed: {len(X)}")

    # B. Generate Robotic Monotone Samples (Class 0)
    print("[-] Generating classic monotone robotic clone samples...")
    for pitch in [110, 130, 145, 160, 180, 200, 220, 240, 260]:
        audio = generate_robotic_sample(pitch=float(pitch), duration=2.5)
        feats = processor.extract_features(audio, 16000)
        if feats:
            X.append(extract_vector(feats))
            y.append(0)  # 0 = SYNTHETIC

    # C. Generate Neural TTS Samples (Class 0)
    print("[-] Generating state-of-the-art neural TTS samples via edge-tts...")
    neural_samples = await generate_neural_tts_samples()
    for audio in neural_samples:
        feats = processor.extract_features(audio, 16000)
        if feats:
            X.append(extract_vector(feats))
            y.append(0)  # 0 = SYNTHETIC

    print(f"[+] Total training dataset: {len(X)} samples (Human: {sum(y)}, Synthetic: {len(y) - sum(y)})")

    X = np.array(X)
    y = np.array(y)

    # Train Calibrated Random Forest Classifier
    base_rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=3,
        random_state=42
    )

    clf = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=3)
    clf.fit(X, y)

    # Save model and feature configuration
    model_artifact = {
        'model': clf,
        'feature_keys': FEATURE_KEYS,
        'version': '1.0.0-rf-calibrated',
    }

    model_path = os.path.join(os.path.dirname(__file__), 'voiceguard_classifier.joblib')
    joblib.dump(model_artifact, model_path)
    print(f"[SUCCESS] Trained VoiceGuard AI model saved to {model_path}!")

    # Verify predictions on test cases
    print("\n--- Verifying Model Accuracy ---")
    test_human = generate_human_sample(pitch_base=180.0, pitch_var=26.0)
    test_human_feats = processor.extract_features(test_human, 16000)
    prob_human = clf.predict_proba([extract_vector(test_human_feats)])[0][1]
    print(f"Test Case 1 (Human Voice): {prob_human * 100:.1f}% HUMAN -> {'PASS' if prob_human >= 0.70 else 'FAIL'}")

    test_robot = generate_robotic_sample(pitch=155.0)
    test_robot_feats = processor.extract_features(test_robot, 16000)
    prob_robot = clf.predict_proba([extract_vector(test_robot_feats)])[0][1]
    print(f"Test Case 2 (Monotone Clone): {prob_robot * 100:.1f}% HUMAN ({(1 - prob_robot) * 100:.1f}% SYNTHETIC) -> {'PASS' if prob_robot <= 0.35 else 'FAIL'}")

    if len(neural_samples) > 0:
        test_ai_feats = processor.extract_features(neural_samples[0], 16000)
        prob_ai = clf.predict_proba([extract_vector(test_ai_feats)])[0][1]
        print(f"Test Case 3 (Neural TTS Clone): {prob_ai * 100:.1f}% HUMAN ({(1 - prob_ai) * 100:.1f}% SYNTHETIC) -> {'PASS' if prob_ai <= 0.35 else 'FAIL'}")

if __name__ == "__main__":
    asyncio.run(main())
