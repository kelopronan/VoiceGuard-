"""
Audio Feature Extraction Engine for VoiceGuard AI
Extracts acoustic features from audio chunks for deepfake detection.
"""

import numpy as np
import base64
import io

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[!] librosa not installed. Install with: pip install librosa")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class AudioProcessor:
    """Processes audio chunks and extracts acoustic features for analysis."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def decode_audio_chunk(self, base64_audio: str) -> np.ndarray | None:
        """Decode base64-encoded raw PCM float32 audio to numpy array."""
        try:
            audio_bytes = base64.b64decode(base64_audio)

            # First try: raw PCM float32 (from browser AudioWorklet/ScriptProcessor)
            if len(audio_bytes) % 4 == 0:
                audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()
                if len(audio) > 0 and np.max(np.abs(audio)) <= 1.5:
                    return audio

            # Second try: decode as wav/webm/ogg via soundfile
            if SOUNDFILE_AVAILABLE:
                try:
                    audio, sr = sf.read(io.BytesIO(audio_bytes))
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    if sr != self.sr and LIBROSA_AVAILABLE:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
                    return audio.astype(np.float32)
                except Exception:
                    pass

            return None
        except Exception as e:
            print(f"Audio decode error: {e}")
            return None

    def extract_features(self, audio: np.ndarray, sr: int = None) -> dict | None:
        """
        Extract comprehensive acoustic features from audio.
        
        Returns dict with:
            - mfcc_mean/std: 13 MFCC coefficients statistics
            - mel_spectrogram: averaged mel-spectrogram bins
            - pitch_mean/std: fundamental frequency statistics (F0 only)
            - energy_rms/std: RMS energy statistics
            - spectral_centroid: brightness of sound
            - spectral_bandwidth: width of spectral energy
            - spectral_flatness: how noise-like the spectrum is
            - zero_crossing_rate: rate of sign changes
            - waveform: downsampled waveform for visualization
        """
        if not LIBROSA_AVAILABLE:
            return self._extract_basic_features(audio)

        if sr is None:
            sr = self.sr

        # Need at least 100ms of audio
        if len(audio) < sr * 0.1:
            return None

        # Ensure float32 and finite
        audio = np.nan_to_num(audio.astype(np.float32))
        if np.max(np.abs(audio)) < 1e-6:
            return None

        features = {}

        try:
            # --- MFCCs (13 coefficients) ---
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.nan_to_num(mfccs.mean(axis=1)).tolist()
            features['mfcc_std'] = np.nan_to_num(mfccs.std(axis=1)).tolist()

            # --- Mel Spectrogram ---
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spectrogram'] = np.nan_to_num(mel_db.mean(axis=1)).tolist()

            # --- Pitch (F0) using piptrack with proper F0 extraction ---
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=60, fmax=500)
            # Extract ONLY the dominant F0 per frame (not harmonics)
            f0_per_frame = []
            for frame_idx in range(pitches.shape[1]):
                frame_pitches = pitches[:, frame_idx]
                frame_mags = magnitudes[:, frame_idx]
                # Get the pitch with highest magnitude in this frame (= F0)
                if np.max(frame_mags) > 0:
                    best_bin = np.argmax(frame_mags)
                    p = frame_pitches[best_bin]
                    if 60 < p < 500:
                        f0_per_frame.append(p)

            if len(f0_per_frame) > 5:
                f0_array = np.array(f0_per_frame)
                features['pitch_mean'] = float(np.mean(f0_array))
                features['pitch_std'] = float(np.std(f0_array))
                # Also compute pitch range and jitter for better discrimination
                features['pitch_range'] = float(np.ptp(f0_array))  # max - min
                # Frame-to-frame jitter (micro-variation)
                if len(f0_array) > 2:
                    diffs = np.abs(np.diff(f0_array))
                    features['pitch_jitter'] = float(np.mean(diffs) / (np.mean(f0_array) + 1e-8))
                else:
                    features['pitch_jitter'] = 0.0
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
                features['pitch_jitter'] = 0.0

            # --- Energy (RMS) with frame-level stats ---
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_rms'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            # Coefficient of variation of energy (normalized dynamics)
            features['energy_cv'] = float(np.std(rms) / (np.mean(rms) + 1e-8))

            # --- Spectral Centroid ---
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))

            # --- Spectral Bandwidth ---
            spectral_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth'] = float(np.mean(spectral_bw))

            # --- Spectral Flatness ---
            spectral_flat = librosa.feature.spectral_flatness(y=audio)[0]
            features['spectral_flatness'] = float(np.mean(spectral_flat))

            # --- Zero Crossing Rate ---
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))

            # --- Spectral Rolloff ---
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff'] = float(np.mean(rolloff))

            # --- Delta MFCC (temporal dynamics) ---
            try:
                delta_mfcc = librosa.feature.delta(mfccs)
                features['delta_mfcc_std'] = np.nan_to_num(delta_mfcc.std(axis=1)).tolist()
            except Exception:
                features['delta_mfcc_std'] = [0.0] * 13

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self._extract_basic_features(audio)

        # --- Downsampled waveform for visualization ---
        downsample_factor = max(1, len(audio) // 200)
        features['waveform'] = audio[::downsample_factor].tolist()[:200]

        return features

    def _extract_basic_features(self, audio: np.ndarray) -> dict | None:
        """Fallback feature extraction without librosa."""
        if len(audio) < 100:
            return None

        features = {}
        features['energy_rms'] = float(np.sqrt(np.mean(audio ** 2)))
        features['energy_std'] = float(np.std(np.abs(audio)))
        features['energy_cv'] = float(features['energy_std'] / (features['energy_rms'] + 1e-8))

        # Basic zero crossing
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        features['zero_crossing_rate'] = float(zero_crossings / len(audio))

        # Basic spectral via FFT
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
        if np.sum(fft) > 0:
            features['spectral_centroid'] = float(np.sum(freqs * fft) / np.sum(fft))
        else:
            features['spectral_centroid'] = 0.0

        features['spectral_flatness'] = float(
            np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10)
        )

        features['pitch_mean'] = 0.0
        features['pitch_std'] = 0.0
        features['pitch_range'] = 0.0
        features['pitch_jitter'] = 0.0
        features['spectral_centroid_std'] = 0.0
        features['spectral_bandwidth'] = 0.0
        features['spectral_rolloff'] = 0.0
        features['mfcc_mean'] = [0.0] * 13
        features['mfcc_std'] = [0.0] * 13
        features['delta_mfcc_std'] = [0.0] * 13
        features['mel_spectrogram'] = [0.0] * 64

        downsample_factor = max(1, len(audio) // 200)
        features['waveform'] = audio[::downsample_factor].tolist()[:200]

        return features
