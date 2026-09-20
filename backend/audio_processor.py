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

        # ─── Device Invariance 1: Sub-bass Rumble & Mains Hum Filter (70 Hz) ───
        # Eliminates 50/60 Hz electrical mains hum, laptop fan vibration, and DC bias
        audio = audio - float(np.mean(audio))
        try:
            from scipy import signal
            b, a = signal.butter(4, 70.0, btype='highpass', fs=sr)
            audio = signal.filtfilt(b, a, audio).astype(np.float32)
        except Exception:
            pass

        # ─── Device Invariance 2: Dynamic Peak AGC Normalization ───
        # Standardizes input level to -1dBFS so quiet budget mics and loud studio mics produce identical scale
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-4:
            audio = (audio / peak * 0.90).astype(np.float32)
        else:
            return None

        # ─── Device Invariance 3: Voice Activity Silence Trimming ───
        # Removes leading/trailing room silence so speech pauses do not distort vocal tract metrics
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=26)
            if len(trimmed) >= int(sr * 0.3):
                audio = trimmed
        except Exception:
            pass

        features = {}

        try:
            # --- MFCCs (13 coefficients) ---
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.nan_to_num(mfccs.mean(axis=1)).tolist()
            features['mfcc_std'] = np.nan_to_num(mfccs.std(axis=1)).tolist()
            # Pure vocal tract shape variation (excluding coefficient 0 which is gain/loudness)
            features['vocal_tract_mfcc_std'] = float(np.mean(mfccs[1:].std(axis=1)))

            # --- Mel Spectrogram ---
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spectrogram'] = np.nan_to_num(mel_db.mean(axis=1)).tolist()

            # --- Fundamental Frequency (F0) using Autocorrelation Pitch Tracking ---
            min_lag = max(1, int(sr / 480))  # 480 Hz (supports high female/child voices)
            max_lag = min(len(audio) // 2, int(sr / 65))   # 65 Hz (supports low male voices)
            frame_len = 1024
            hop_len = 256
            f0_vals = []

            for i in range(0, len(audio) - frame_len, hop_len):
                frame = audio[i:i + frame_len]
                # Normalized audio: voiced speech frames have std >= 0.025
                if np.std(frame) < 0.025:
                    continue
                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(frame) - 1:]
                search_win = corr[min_lag:max_lag]
                if len(search_win) > 0:
                    peak_lag = min_lag + int(np.argmax(search_win))
                    norm_peak = float(corr[peak_lag] / (corr[0] + 1e-8))
                    if norm_peak > 0.30:
                        freq = float(sr / peak_lag)
                        if 65 <= freq <= 480:
                            f0_vals.append(freq)

            if len(f0_vals) >= 3:
                f0_array = np.array(f0_vals)
                p_mean = float(np.mean(f0_array))
                p_std = float(np.std(f0_array))
                features['pitch_mean'] = p_mean
                features['pitch_std'] = p_std
                features['pitch_range'] = float(np.ptp(f0_array))
                # Scale-invariant Relative Pitch CV (%)
                features['pitch_cv_percent'] = float((p_std / (p_mean + 1e-8)) * 100.0)
                if len(f0_array) > 2:
                    diffs = np.abs(np.diff(f0_array))
                    features['pitch_jitter'] = float(np.mean(diffs) / (p_mean + 1e-8))
                else:
                    features['pitch_jitter'] = 0.0
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
                features['pitch_cv_percent'] = 0.0
                features['pitch_jitter'] = 0.0

            # --- Energy (RMS) with frame-level stats ---
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_rms'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_cv'] = float(np.std(rms) / (np.mean(rms) + 1e-8))

            # --- Spectral Centroid ---
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))

            # --- Spectral Bandwidth ---
            spectral_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth'] = float(np.mean(spectral_bw))

            # --- Device Invariance 3: Speech Formant-Band Flatness (200Hz - 3800Hz) ---
            # Measures true glottal resonance in human speech band, immune to laptop fan hiss or ultrasonic noise
            S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
            fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            speech_mask = (fft_freqs >= 200) & (fft_freqs <= 3800)
            S_speech = S[speech_mask, :]
            geo_m = np.exp(np.mean(np.log(S_speech + 1e-12), axis=0))
            ari_m = np.mean(S_speech, axis=0) + 1e-12
            features['spectral_flatness'] = float(np.mean(geo_m / ari_m))

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
