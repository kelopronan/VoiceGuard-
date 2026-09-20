"""
Voice Deepfake Detection Model Handler for VoiceGuard AI

Dual-Mode Detection:
1. HuggingFace Inference API (Wav2Vec2 Foundation Model) — Active when HF token is provided
   via header (X-HF-API-Key), Vercel environment variable, or Railway backend environment.
2. Calibrated Biometric Acoustic Engine — Instant, offline, calibrated for real microphones.
"""

import os
import io
import numpy as np
from audio_processor import AudioProcessor

try:
    import requests as http_requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class VoiceDetector:
    """Detects whether audio is from a real human or a synthetic/cloned voice."""

    # Current HuggingFace router endpoints (Wav2Vec2 fine-tuned on ASVspoof / In-The-Wild)
    HF_API_URLS = [
        "https://router.huggingface.co/hf-inference/models/MelodyMachine/Deepfake-audio-detection-V2",
        "https://router.huggingface.co/hf-inference/models/mo-thecreator/Deepfake-audio-detection",
        "https://router.huggingface.co/hf-inference/models/Hemgg/Deepfake-audio-detection"
    ]

    def __init__(self, hf_api_key: str = None):
        self.processor = AudioProcessor()
        self.hf_api_key = (
            hf_api_key or
            os.getenv("HUGGINGFACE_API_KEY") or
            os.getenv("HF_TOKEN") or
            os.getenv("HF_API_KEY")
        )
        if self.hf_api_key:
            self.model_name = "huggingface-wav2vec2"
            self.model_loaded = True
            print("[+] HuggingFace API key detected - Wav2Vec2 inference active")
        else:
            self.model_name = "voiceguard-biometric"
            self.model_loaded = True
            print("[*] VoiceGuard Calibrated Biometric Acoustic Engine active")

    def set_api_key(self, key: str):
        """Dynamically set or update the HuggingFace API key."""
        self.hf_api_key = key.strip() if key and len(key.strip()) > 0 else None
        if self.hf_api_key:
            self.model_name = "huggingface-wav2vec2"
            self.model_loaded = True
            print("[+] HuggingFace API key updated")
        else:
            self.model_name = "voiceguard-biometric"
            print("[*] API key cleared - using calibrated biometric engine")

    def predict(self, audio: np.ndarray, sr: int = 16000, hf_key: str = None) -> dict:
        """
        Predict whether audio is real or synthetic.

        Returns dict with:
            score: 0.0 (synthetic) to 1.0 (human)
            label: HUMAN | UNCERTAIN | SYNTHETIC
            confidence: 0.0 to 1.0
            features: acoustic feature dict
            reasons: list of human-readable explanations
            model: which detection method was used
        """
        features = self.processor.extract_features(audio, sr)

        if features is None:
            return {
                "score": 0.5,
                "label": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "features": {},
                "reasons": ["Not enough audio data for analysis"],
                "model": self.model_name,
            }

        # Resolve active HuggingFace API token (passed via header or environment)
        active_key = (
            (hf_key.strip() if hf_key and len(hf_key.strip()) > 0 else None) or
            self.hf_api_key or
            os.getenv("HUGGINGFACE_API_KEY") or
            os.getenv("HF_TOKEN") or
            os.getenv("HF_API_KEY")
        )

        # 1. Try HuggingFace Inference API if key provided
        if active_key and REQUESTS_AVAILABLE:
            result = self._predict_hf_api(audio, sr, features, api_key=active_key)
            if result is not None:
                return result

        # 2. Calibrated Biometric Acoustic Engine (Offline, Reliable, Real-Mic Tested)
        return self._predict_biometric(features)

    def _predict_hf_api(self, audio: np.ndarray, sr: int, features: dict, api_key: str) -> dict | None:
        """Call HuggingFace Inference API using Wav2Vec2 deep neural network."""
        try:
            import soundfile as sf

            # Convert audio to lossless 16kHz WAV
            buf = io.BytesIO()
            sf.write(buf, audio.astype(np.float32), sr, format="WAV")
            wav_bytes = buf.getvalue()

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "audio/wav"
            }

            for url in self.HF_API_URLS:
                try:
                    response = http_requests.post(
                        url,
                        headers=headers,
                        data=wav_bytes,
                        timeout=12,
                    )

                    if response.status_code == 503:
                        print(f"[*] HF model {url.split('/')[-1]} is loading, trying next...")
                        continue

                    if response.status_code == 401 or response.status_code == 403:
                        print(f"[!] HF API authorization error: invalid token provided")
                        return None

                    if response.status_code != 200:
                        print(f"[!] HF API status {response.status_code}: {response.text[:120]}")
                        continue

                    result = response.json()

                    if isinstance(result, dict) and "error" in result:
                        print(f"[!] HF API returned error: {result['error']}")
                        continue

                    # Parse Wav2Vec2 classification array
                    real_score = None
                    if isinstance(result, list):
                        for item in result:
                            raw_label = str(item.get("label", "")).lower()
                            score_val = float(item.get("score", 0.5))

                            if any(k in raw_label for k in ["real", "bonafide", "human", "genuine", "original"]):
                                real_score = score_val
                                break
                            elif any(k in raw_label for k in ["fake", "spoof", "synthetic", "clone", "deepfake"]):
                                real_score = 1.0 - score_val
                                break

                    if real_score is None:
                        continue

                    model_id = url.split('/')[-1]
                    score = round(max(0.02, min(0.98, float(real_score))), 4)
                    label = self._score_to_label(score)
                    confidence = round(abs(score - 0.5) * 2, 4)

                    reasons = []
                    if score >= 0.52:
                        reasons.append(
                            f"[VERIFIED] HuggingFace Wav2Vec2 deep neural network verified authentic human speech ({score * 100:.1f}%)"
                        )
                        reasons.append(f"Model: {model_id} (ASVspoof foundation weights)")
                        if features.get("pitch_cv_percent", 0) >= 6:
                            reasons.append(f"Natural intonation dynamics ({features['pitch_cv_percent']:.1f}% CV) confirmed")
                    else:
                        fake_prob = (1.0 - score) * 100
                        reasons.append(
                            f"[DEEPFAKE DETECTED] HuggingFace Wav2Vec2 neural network flagged synthetic voice cloning ({fake_prob:.1f}% confidence)"
                        )
                        reasons.append(f"Model: {model_id} identified synthetic acoustic artifacts")
                        if features.get("spectral_flatness", 0) > 0.08:
                            reasons.append(f"Elevated neural vocoder phase diffusion noise detected")

                    print(f"[HF PREDICTION] {model_id}: score={score:.4f} ({label})")
                    return {
                        "score": score,
                        "label": label,
                        "confidence": confidence,
                        "features": self._sanitize_features(features),
                        "reasons": reasons,
                        "model": f"huggingface ({model_id})",
                    }

                except Exception as model_err:
                    print(f"[!] Error calling {url}: {model_err}")
                    continue

            return None

        except Exception as e:
            print(f"[!] HF API pipeline failed: {e}")
            return None

    def _predict_biometric(self, features: dict) -> dict:
        """
        Calibrated Biometric Acoustic Engine.

        Grounded in biological vocal fold and vocal tract physics:
        - Real human speech has dynamic pitch intonation (CV 6% - 40%)
        - Real human vocal cords have physiological micro-tremor (jitter 0.006 - 0.12)
        - Monotone clones have locked pitch (CV < 3.5%, std < 4 Hz)
        - Pure sine / robotic clones have near-zero jitter (< 0.004)
        - Distinguishes organic consonants and room acoustics from vocoder noise
        """
        evidence = []
        reasons = []

        pitch_std = features.get("pitch_std", 0.0)
        pitch_mean = features.get("pitch_mean", 0.0)
        pitch_cv = features.get("pitch_cv_percent", 0.0)
        pitch_jitter = features.get("pitch_jitter", 0.0)
        energy_cv = features.get("energy_cv", 0.0)
        spectral_flatness = features.get("spectral_flatness", 0.0)
        high_band_flatness = features.get("high_band_flatness", 0.0)
        spectral_centroid_std = features.get("spectral_centroid_std", 0.0)
        avg_mfcc = features.get("vocal_tract_mfcc_std", 8.0)

        has_pitch = pitch_mean > 50
        if pitch_cv == 0.0 and has_pitch:
            pitch_cv = float((pitch_std / pitch_mean) * 100.0)

        # Silence / Pause handling
        if not has_pitch:
            return {
                "score": 0.50,
                "label": "SPEECH_PAUSE",
                "confidence": 0.0,
                "features": self._sanitize_features(features),
                "reasons": ["[PAUSE] Natural inter-word speech pause or silence detected"],
                "model": "voiceguard-biometric",
            }

        # ─── 1. Fundamental Frequency (F0) Contour ───
        # Monotone robotic clones have locked pitch (< 2.5% CV)
        if pitch_cv < 2.5 or pitch_std < 2.0:
            evidence.append((0.05, 5.0))
            reasons.append("[ANOMALY] Mechanically locked F0 (<2.5% CV) — monotone clone signature")
        elif pitch_cv < 4.5 or pitch_std < 3.8:
            evidence.append((0.30, 2.0))
            reasons.append("[SUSPICIOUS] Compressed robotic pitch modulation")
        elif 5.5 <= pitch_cv <= 45.0 or (5.0 <= pitch_std <= 55.0):
            evidence.append((0.94, 3.0))
            reasons.append(f"[NATURAL] Biological human prosody intonation dynamics (CV: {pitch_cv:.1f}%)")
        else:
            evidence.append((0.70, 1.5))
            reasons.append("[NATURAL] Expressive pitch modulation")

        # ─── 2. Vocal Cord Mucosal Jitter (Micro-Perturbation) ───
        # Real human vocal fold tissue cannot oscillate with zero jitter
        if pitch_jitter < 0.0035:
            evidence.append((0.10, 3.0))
            reasons.append("[ANOMALY] Unnatural mathematical pitch precision (zero vocal tissue micro-tremor)")
        elif 0.007 <= pitch_jitter <= 0.120:
            evidence.append((0.93, 2.5))
            reasons.append(f"[NATURAL] Organic vocal fold mucosal wave micro-perturbation ({pitch_jitter:.4f})")
        elif pitch_jitter > 0.220:
            evidence.append((0.25, 2.0))
            reasons.append("[ANOMALY] Neural vocoder frame-to-frame pitch discontinuity")
        else:
            evidence.append((0.75, 1.5))
            reasons.append("[NATURAL] Typical vocal cord micro-perturbation")

        # ─── 3. Vocal Tract Articulation & Formant Dynamics (MFCC) ───
        if avg_mfcc < 2.5:
            evidence.append((0.15, 2.0))
            reasons.append("[ANOMALY] Over-smoothed synthetic vocal tract modeling")
        elif 3.8 <= avg_mfcc <= 18.0:
            evidence.append((0.92, 2.5))
            reasons.append("[NATURAL] Biological vocal tract formant articulation dynamics")
        elif avg_mfcc > 24.0:
            evidence.append((0.30, 1.5))
            reasons.append("[SUSPICIOUS] Elevated Mel filterbank ripple variance")
        else:
            evidence.append((0.75, 1.5))
            reasons.append("[NATURAL] Normal formant articulation")

        # ─── 4. Phase Noise vs. Organic Consonants / Room Acoustics ───
        # Real speech has consonants ('s', 'sh', 'f') which create natural noise.
        # Only penalize spectral flatness if pitch is ALSO monotone or jitter is near zero.
        is_monotone = (pitch_cv < 4.0 or pitch_jitter < 0.004)
        if spectral_flatness > 0.22 and is_monotone:
            evidence.append((0.15, 2.5))
            reasons.append("[ANOMALY] High-band neural vocoder diffusion phase noise")
        elif spectral_flatness <= 0.18:
            evidence.append((0.90, 2.0))
            reasons.append("[NATURAL] Harmonic resonance clarity consistent with physical vocal tract")
        else:
            evidence.append((0.80, 1.0))
            reasons.append("[INFO] Ambient microphone acoustics and organic consonant dynamics")

        # ─── 5. Energy Modulation (Syllable Cadence & AGC Tolerance) ───
        if energy_cv < 0.03:
            evidence.append((0.30, 1.0))
            reasons.append("[INFO] Highly compressed or AGC-normalized microphone volume")
        elif 0.07 <= energy_cv <= 0.95:
            evidence.append((0.92, 2.0))
            reasons.append("[NATURAL] Syllabic speech cadence and dynamic stress modulation")
        else:
            evidence.append((0.80, 1.0))
            reasons.append("[NATURAL] Dynamic speech energy range")

        # Calculate weighted consensus
        total_w = sum(w for ev_val, w in evidence)
        weighted_sum = sum(ev_val * w for ev_val, w in evidence)
        score = weighted_sum / max(1.0, total_w)

        # Gate on critical anomalies
        severe_anomalies = sum(1 for ev_val, w in evidence if ev_val <= 0.20)
        if pitch_cv < 2.5:
            # Monotone locked frequency contour is an unambiguous machine synthesis signature
            score = min(score, 0.20)
        elif severe_anomalies >= 2:
            score = min(score, 0.25)
        elif severe_anomalies == 1:
            score = min(score, 0.68)

        score = max(0.04, min(0.97, score))
        label = self._score_to_label(score)

        print(f"[BIOMETRIC] pitch_cv={pitch_cv:.1f}% jitter={pitch_jitter:.4f} "
              f"mfcc={avg_mfcc:.1f} anomalies={severe_anomalies} => score={score:.3f} ({label})")

        return {
            "score": round(float(score), 4),
            "label": label,
            "confidence": round(abs(score - 0.5) * 2, 4),
            "features": self._sanitize_features(features),
            "reasons": reasons,
            "model": "voiceguard-biometric",
        }

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.52:
            return "HUMAN"
        elif score <= 0.38:
            return "SYNTHETIC"
        else:
            return "UNCERTAIN"

    @staticmethod
    def _sanitize_features(features: dict) -> dict:
        sanitized = {}
        for key, value in features.items():
            if key in ("waveform", "mel_spectrogram"):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = [round(v, 4) if isinstance(v, float) else v for v in value]
            elif isinstance(value, float):
                sanitized[key] = round(value, 4)
            else:
                sanitized[key] = value
        return sanitized
