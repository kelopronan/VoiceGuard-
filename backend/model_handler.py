"""
Voice Deepfake Detection Model Handler for VoiceGuard AI

Three detection modes (auto-selected):
1. HuggingFace Inference API — uses API key, no local model (best for hackathon)
2. Local HuggingFace model — needs transformers + torch installed
3. Heuristic — acoustic feature analysis, always available, zero downloads
"""

import numpy as np
import io
from audio_processor import AudioProcessor

try:
    import requests as http_requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class VoiceDetector:
    """Detects whether audio is from a real human or a synthetic/cloned voice."""

    HF_API_URL = "https://api-inference.huggingface.co/models/motheecreator/Deepfake-audio-detection"

    def __init__(self, hf_api_key: str = None):
        self.processor = AudioProcessor()
        self.hf_api_key = hf_api_key
        self.model_loaded = False
        self.model_name = "heuristic-v2"

        if hf_api_key:
            self.model_name = "huggingface-api"
            self.model_loaded = True
            print("[+] HuggingFace API key set - using Inference API")
        else:
            print("[*] Using heuristic-based detection v2 (no API key set)")

    def set_api_key(self, key: str):
        """Dynamically set/update the HuggingFace API key."""
        self.hf_api_key = key if key and len(key.strip()) > 0 else None
        if self.hf_api_key:
            self.model_name = "huggingface-api"
            self.model_loaded = True
            print("[+] HuggingFace API key updated")
        else:
            self.model_name = "heuristic-v2"
            self.model_loaded = False
            print("[*] API key cleared - using heuristic detection")

    def predict(self, audio: np.ndarray, sr: int = 16000) -> dict:
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

        # Try HuggingFace Inference API first
        if self.hf_api_key and REQUESTS_AVAILABLE:
            result = self._predict_hf_api(audio, sr, features)
            if result is not None:
                return result

        # Fallback to heuristic
        return self._predict_heuristic(features)

    def _predict_hf_api(self, audio: np.ndarray, sr: int, features: dict) -> dict | None:
        """Call HuggingFace Inference API for deepfake detection."""
        try:
            import soundfile as sf

            # Convert audio to WAV bytes
            buf = io.BytesIO()
            sf.write(buf, audio.astype(np.float32), sr, format="WAV")
            wav_bytes = buf.getvalue()

            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            response = http_requests.post(
                self.HF_API_URL,
                headers=headers,
                data=wav_bytes,
                timeout=10,
            )

            if response.status_code == 503:
                # Model is loading
                print("[*] HF model is loading, falling back to heuristic")
                return None

            if response.status_code != 200:
                print(f"[!] HF API error {response.status_code}: {response.text[:100]}")
                return None

            result = response.json()

            if isinstance(result, dict) and "error" in result:
                print(f"[!] HF API error: {result['error']}")
                return None

            # Parse classification result
            real_score = 0.5
            if isinstance(result, list):
                for item in result:
                    label = item.get("label", "").lower()
                    if any(k in label for k in ["real", "bonafide", "human", "genuine", "original"]):
                        real_score = item["score"]
                        break
                    elif any(k in label for k in ["fake", "spoof", "synthetic", "clone", "deepfake"]):
                        real_score = 1.0 - item["score"]
                        break

            label = self._score_to_label(real_score)
            return {
                "score": round(float(real_score), 4),
                "label": label,
                "confidence": round(abs(real_score - 0.5) * 2, 4),
                "features": self._sanitize_features(features),
                "reasons": [f"HuggingFace API ({self.HF_API_URL.split('/')[-1]})"],
                "model": "huggingface-api",
            }

        except Exception as e:
            print(f"[!] HF API call failed: {e}")
            return None

    def _predict_heuristic(self, features: dict) -> dict:
        """
        Physiological Bounding (Goldilocks) Deepfake Detection Engine.

        Real human vocal physiology operates within bounded physical constraints:
        - Natural conversational F0 standard deviation: 10 - 52 Hz
        - Organic phonemic MFCC variance: 5.0 - 18.0
        - Natural harmonic-to-noise spectral flatness: 0.005 - 0.038
        - Natural respiratory/syllable dynamic energy modulation: 0.18 - 0.85
        - Vocal cord tissue micro-perturbation jitter: 0.015 - 0.12

        Detects BOTH:
        1. Legacy/robotic clones (too low: monotone, flat amplitude, over-smoothed MFCCs)
        2. Modern neural TTS (Gemini, ElevenLabs, VITS, HiFi-GAN):
           - Excessive pitch jumping / vocoder octave dispersion (pitch_std > 65 Hz)
           - Vocoder high-band phase noise (flatness > 0.05)
           - Hyper-articulated synthetic mel ripple (mfcc_avg > 21)
           - Extreme frame-to-frame pitch discontinuity (jitter > 0.16)
        """
        evidence = []
        reasons = []

        pitch_std = features.get("pitch_std", 0.0)
        pitch_mean = features.get("pitch_mean", 0.0)
        pitch_jitter = features.get("pitch_jitter", 0.0)
        energy_cv = features.get("energy_cv", 0.0)
        spectral_flatness = features.get("spectral_flatness", 0.0)
        spectral_centroid_std = features.get("spectral_centroid_std", 0.0)
        mfcc_std = features.get("mfcc_std", [0] * 13)
        delta_mfcc_std = features.get("delta_mfcc_std", [0] * 13)

        avg_mfcc = float(np.mean(mfcc_std)) if isinstance(mfcc_std, list) and len(mfcc_std) > 0 else 0.0
        has_pitch = pitch_mean > 50

        # ─── 1. Fundamental Frequency (F0) Dynamics (Goldilocks: 10 - 52 Hz) ───
        if has_pitch:
            if pitch_std < 4.0:
                evidence.append(0.06)
                reasons.append("[ANOMALY] Mechanically locked F0 - monotone clone signature")
            elif pitch_std < 9.0:
                evidence.append(0.25)
                reasons.append("[SUSPICIOUS] Compressed robotic pitch modulation")
            elif 10.0 <= pitch_std <= 52.0:
                evidence.append(0.92)
                reasons.append("[NATURAL] Fundamental frequency within human vocal cord range (10-52 Hz)")
            elif 52.0 < pitch_std <= 68.0:
                evidence.append(0.55)
                reasons.append("[NATURAL] Wide expressive pitch intonation")
            else:
                # > 68 Hz in 3s window = neural vocoder octave dispersion / phase jumping
                evidence.append(0.12)
                reasons.append("[ANOMALY] Neural vocoder phase jumps & erratic octave dispersion (>68 Hz)")
        else:
            evidence.append(0.50)
            reasons.append("[INFO] Low voiced signal for F0 tracking")

        # ─── 2. MFCC Articulatory Complexity (Goldilocks: 5.0 - 18.0) ───
        if avg_mfcc < 3.0:
            evidence.append(0.10)
            reasons.append("[ANOMALY] Over-smoothed acoustic vocal tract modeling")
        elif avg_mfcc < 5.0:
            evidence.append(0.35)
            reasons.append("[SUSPICIOUS] Sub-normal articulatory diversity")
        elif 5.0 <= avg_mfcc <= 18.0:
            evidence.append(0.92)
            reasons.append("[NATURAL] Organic vocal tract formant articulation dynamics")
        elif 18.0 < avg_mfcc <= 21.0:
            evidence.append(0.55)
            reasons.append("[INFO] High articulatory acoustic variance")
        else:
            # > 21.0 = neural synthesis spectral ripple / mel dispersion artifact
            evidence.append(0.14)
            reasons.append("[ANOMALY] Neural vocoder spectral ripple & synthetic mel dispersion (>21)")

        # ─── 3. Spectral Flatness / Wiener Entropy (Goldilocks: 0.005 - 0.038) ───
        if spectral_flatness < 0.0005:
            evidence.append(0.10)
            reasons.append("[ANOMALY] Mathematically pure synthetic harmonics (zero glottal turbulence)")
        elif 0.005 <= spectral_flatness <= 0.038:
            evidence.append(0.90)
            reasons.append("[NATURAL] Natural harmonic formant peaks with organic air turbulence")
        elif 0.038 < spectral_flatness <= 0.052:
            evidence.append(0.50)
            reasons.append("[INFO] Moderate background noise / fricative energy")
        else:
            # > 0.052 = neural vocoder diffusion / GAN generator phase noise
            evidence.append(0.12)
            reasons.append("[ANOMALY] Neural vocoder high-band phase noise signature (>0.05)")

        # ─── 4. Energy Modulation (Goldilocks: 0.18 - 0.85) ───
        if energy_cv < 0.06:
            evidence.append(0.08)
            reasons.append("[ANOMALY] Flat unmodulated machine amplitude envelope")
        elif energy_cv < 0.15:
            evidence.append(0.30)
            reasons.append("[SUSPICIOUS] Compressed syllable stress dynamics")
        elif 0.18 <= energy_cv <= 0.85:
            evidence.append(0.92)
            reasons.append("[NATURAL] Organic syllable stress and respiratory breathing pauses")
        else:
            evidence.append(0.40)
            reasons.append("[SUSPICIOUS] Non-biological energy envelope gating")

        # ─── 5. Micro-Jitter / Pitch Perturbation (Goldilocks: 0.015 - 0.12) ───
        if has_pitch:
            if pitch_jitter < 0.005:
                evidence.append(0.10)
                reasons.append("[ANOMALY] Zero micro-jitter - unnaturally perfect pitch synthesis")
            elif 0.015 <= pitch_jitter <= 0.12:
                evidence.append(0.88)
                reasons.append("[NATURAL] Natural laryngeal vocal cord micro-perturbations")
            elif pitch_jitter > 0.16:
                # Extreme jitter = vocoder phase discontinuity / octave leaping
                evidence.append(0.15)
                reasons.append("[ANOMALY] Severe frame-to-frame pitch discontinuity (vocoder tracking jump)")
            else:
                evidence.append(0.55)
        else:
            evidence.append(0.50)

        # ─── 6. Spectral Centroid Dynamic Variance ───
        if spectral_centroid_std < 70.0:
            evidence.append(0.15)
            reasons.append("[ANOMALY] Stationary spectral brightness - synthetic static vocal tract")
        elif spectral_centroid_std >= 180.0:
            evidence.append(0.85)
            reasons.append("[NATURAL] Dynamic vowel formant transitions across speech frames")
        else:
            evidence.append(0.55)

        # ─── Weights: [Pitch F0, MFCC, Flatness, Energy, Jitter, Centroid] ───
        weights = [2.5, 2.0, 2.0, 1.5, 1.5, 1.0]
        assert len(evidence) == len(weights)

        weighted_sum = sum(e * w for e, w in zip(evidence, weights))
        score = weighted_sum / sum(weights)

        # ─── Neural Deepfake Anomaly Decision Gate ───
        # In acoustic forensics, multiple independent vocoder anomalies (e.g. high-band noise,
        # unnatural F0 dispersion, extreme jitter) cannot occur simultaneously in human biology.
        anomaly_count = sum(1 for r in reasons if "[ANOMALY]" in r)
        if anomaly_count >= 3:
            score = min(score, 0.22)
        elif anomaly_count >= 2:
            score = min(score, 0.35)
        elif anomaly_count == 1:
            score = min(score, 0.58)

        score = max(0.04, min(0.97, score))
        label = self._score_to_label(score)

        print(f"[FORENSIC] pitch_std={pitch_std:.1f} mfcc={avg_mfcc:.1f} "
              f"flatness={spectral_flatness:.4f} jitter={pitch_jitter:.4f} "
              f"anomalies={anomaly_count} => score={score:.3f} ({label})")

        return {
            "score": round(float(score), 4),
            "label": label,
            "confidence": round(abs(score - 0.5) * 2, 4),
            "features": self._sanitize_features(features),
            "reasons": reasons,
            "model": "heuristic-v2",
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
