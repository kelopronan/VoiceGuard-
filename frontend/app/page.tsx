'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Shield, ShieldCheck, ShieldAlert, Mic, Square, Upload, Play, Pause, Download,
  Activity, BarChart3, Tags, Zap, Bot, Package,
  AlertTriangle, CheckCircle, XCircle, Sun, Moon,
  FileAudio, History, Clock, Wifi, WifiOff, Volume2, X, Sparkles,
  Radio, FileText, Gauge, RefreshCw, AudioWaveform, Key
} from 'lucide-react';
import TrustGauge from '@/components/TrustGauge';
import WaveformCanvas from '@/components/WaveformCanvas';

/* ─── Types ───────────────────────────────────────────────────────────────── */
interface AnalysisResult {
  score?: number;
  label: string;
  confidence?: number;
  features?: {
    pitch_mean?: number;
    pitch_std?: number;
    energy_rms?: number;
    energy_std?: number;
    spectral_centroid?: number;
    spectral_flatness?: number;
    zero_crossing_rate?: number;
    mfcc_mean?: number[];
    mfcc_std?: number[];
    waveform?: number[];
  };
  reasons?: string[];
  model?: string;
  inference_ms?: number;
  buffer_seconds?: number;
  timestamp?: number;
  filename?: string;
  duration?: number;
  error?: string;
}

interface HistoryEntry extends AnalysisResult {
  time: string;
}

/* ─── Constants ───────────────────────────────────────────────────────────── */
let rawBackend = (process.env.NEXT_PUBLIC_BACKEND_URL || '').trim();
if (rawBackend && !rawBackend.startsWith('http://') && !rawBackend.startsWith('https://')) {
  rawBackend = `https://${rawBackend}`;
}
const BACKEND_URL = rawBackend.replace(/\/+$/, '');
const API_BASE = BACKEND_URL ? `${BACKEND_URL}/api` : '/api';

function downsample(buffer: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    result[i] = buffer[Math.round(i * ratio)];
  }
  return result;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = (seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}


function encodeWAV(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Main Dashboard
   ═══════════════════════════════════════════════════════════════════════════ */
export default function VoiceGuardPage() {
  const [darkMode, setDarkMode] = useState(true);
  const [themeReady, setThemeReady] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('voiceguard-theme');
    const isDark = saved === 'dark' || saved === null;
    setDarkMode(isDark);
    document.documentElement.classList.toggle('dark', isDark);
    setThemeReady(true);
  }, []);

  const toggleDarkMode = () => {
    const next = !darkMode;
    setDarkMode(next);
    document.documentElement.classList.toggle('dark', next);
    localStorage.setItem('voiceguard-theme', next ? 'dark' : 'light');
  };

  /* ─── State ─── */
  const [isRecording, setIsRecording] = useState(false);
  const [trustScore, setTrustScore] = useState<number | null>(null);
  const [label, setLabel] = useState('WAITING');
  const [reasons, setReasons] = useState<string[]>([]);
  const [features, setFeatures] = useState<AnalysisResult['features']>({});
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [recordingTime, setRecordingTime] = useState(0);
  const [inferenceMs, setInferenceMs] = useState(0);
  const [bufferSec, setBufferSec] = useState(0);
  const [modelName, setModelName] = useState('heuristic-v2');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasRecording, setHasRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isAnalyzingRecord, setIsAnalyzingRecord] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [visualMode, setVisualMode] = useState<'waveform' | 'spectrum'>('waveform');

  /* ─── HuggingFace Integration State ─── */
  const [hfKey, setHfKey] = useState('');
  const [showHfModal, setShowHfModal] = useState(false);
  const [hfInput, setHfInput] = useState('');
  const [showTerms, setShowTerms] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('voiceguard_hf_key') || process.env.NEXT_PUBLIC_HF_API_KEY || '';
      if (stored) {
        setHfKey(stored);
        setHfInput(stored);
      }
    }
  }, []);

  const saveHfKey = () => {
    const trimmed = hfInput.trim();
    setHfKey(trimmed);
    if (typeof window !== 'undefined') {
      if (trimmed) {
        localStorage.setItem('voiceguard_hf_key', trimmed);
      } else {
        localStorage.removeItem('voiceguard_hf_key');
      }
    }
    setShowHfModal(false);
  };

  /* ─── Refs ─── */
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const allPcmChunksRef = useRef<Float32Array[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  useEffect(() => {
    return () => {
      stopRecording();
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ─── Analysis Request ─── */
  const analyzeBlob = async (blob: Blob, name: string = 'audio.wav') => {
    setIsAnalyzingRecord(true);
    try {
      const formData = new FormData();
      formData.append('file', blob, name);
      const headers: Record<string, string> = {};
      const activeKey = hfKey || (typeof window !== 'undefined' ? localStorage.getItem('voiceguard_hf_key') : '') || process.env.NEXT_PUBLIC_HF_API_KEY || '';
      if (activeKey) headers['X-HF-API-Key'] = activeKey;
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', headers, body: formData });
      const result: AnalysisResult = await res.json();

      if (result.error) {
        setLabel('ERROR');
        setReasons([result.error]);
        return;
      }

      if (typeof result.score === 'number') {
        setTrustScore(result.score);
        setLabel(result.label || 'UNCERTAIN');
        setReasons(result.reasons || []);
        setFeatures(result.features || {});
        setModelName(result.model || 'heuristic-v2');
        setInferenceMs(result.inference_ms || 0);
        addToHistory(result);
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setLabel('ERROR');
      setReasons(['Analysis failed — backend server unreachable on port 8000']);
    } finally {
      setIsAnalyzingRecord(false);
    }
  };


  /* ─── Recording ─── */
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      setAnalyserNode(analyser);

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      source.connect(processor);
      processor.connect(audioCtx.destination);
      processorRef.current = processor;

      allPcmChunksRef.current = [];

      processor.onaudioprocess = (e) => {
        const data = e.inputBuffer.getChannelData(0);
        allPcmChunksRef.current.push(new Float32Array(data));
      };

      setRecordingTime(0);
      timerRef.current = setInterval(() => setRecordingTime((t) => t + 1), 1000);

      setIsRecording(true);
      setHasRecording(false);
      setRecordedBlob(null);
      setTrustScore(null);
      setLabel('RECORDING');
      setReasons(['Listening... Speak naturally. Click Stop when finished to generate forensic verdict.']);
    } catch (err) {
      console.error('Mic error:', err);
      alert('Microphone access blocked. Please enable microphone permissions in your browser.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }

    const currentSampleRate = audioCtxRef.current?.sampleRate || 16000;

    // Convert accumulated pristine PCM to lossless 16kHz WAV
    const totalLen = allPcmChunksRef.current.reduce((s, a) => s + a.length, 0);
    if (totalLen > 0) {
      const combined = new Float32Array(totalLen);
      let off = 0;
      for (const c of allPcmChunksRef.current) { combined.set(c, off); off += c.length; }
      const resampled = downsample(combined, currentSampleRate, 16000);
      const wavBytes = encodeWAV(resampled, 16000);
      const wavBlob = new Blob([wavBytes], { type: 'audio/wav' });
      setRecordedBlob(wavBlob);
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = URL.createObjectURL(wavBlob);
      setHasRecording(true);

      // Transition immediately to analyzing state
      setLabel('ANALYZING');
      setReasons(['Analyzing complete audio sample across 6-pillar forensic acoustic battery...']);

      // Execute authoritative forensic analysis on complete lossless WAV
      analyzeBlob(wavBlob, 'mic_capture.wav');
    }

    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }

    setAnalyserNode(null);
    allPcmChunksRef.current = [];
    setIsRecording(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ─── Playback ─── */
  const playRecording = () => {
    if (!blobUrlRef.current || !audioElRef.current) return;
    audioElRef.current.src = blobUrlRef.current;
    audioElRef.current.play();
    setIsPlaying(true);
  };

  const pausePlayback = () => {
    audioElRef.current?.pause();
    setIsPlaying(false);
  };

  const downloadRecording = () => {
    if (!blobUrlRef.current) return;
    const a = document.createElement('a');
    a.href = blobUrlRef.current;
    a.download = `voiceguard-telemetry-${Date.now()}.webm`;
    a.click();
  };

  /* ─── File Upload ─── */
  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    analyzeFile(file);
  };

  const analyzeFile = async (fileToAnalyze?: File) => {
    const targetFile = fileToAnalyze || selectedFile;
    if (!targetFile) return;
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', targetFile);
      const headers: Record<string, string> = {};
      const activeKey = hfKey || (typeof window !== 'undefined' ? localStorage.getItem('voiceguard_hf_key') : '') || process.env.NEXT_PUBLIC_HF_API_KEY || '';
      if (activeKey) headers['X-HF-API-Key'] = activeKey;
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', headers, body: formData });
      const result: AnalysisResult = await res.json();

      if (result.error) {
        setLabel('ERROR');
        setReasons([result.error]);
        return;
      }

      if (typeof result.score === 'number') {
        setTrustScore(result.score);
        setLabel(result.label || 'UNCERTAIN');
        setReasons(result.reasons || []);
        setFeatures(result.features || {});
        setModelName(result.model || 'heuristic-v2');
        setInferenceMs(result.inference_ms || 0);
        addToHistory(result);
      }
    } catch (err) {
      console.error('Upload error:', err);
      setLabel('ERROR');
      setReasons(['Backend server unavailable at http://localhost:8000']);
    } finally {
      setIsUploading(false);
    }
  };

  /* ─── Export Forensic JSON Report ─── */
  const exportReport = () => {
    const reportData = {
      product: 'VoiceGuard AI Acoustic Forensics',
      generatedAt: new Date().toISOString(),
      currentVerdict: label,
      trustScore: trustScore,
      extractedFeatures: features,
      reasons: reasons,
      sessionCount: history.length,
      auditLog: history,
    };
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(reportData, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `voiceguard-audit-${Date.now()}.json`;
    a.click();
  };

  const addToHistory = (result: AnalysisResult) => {
    const entry: HistoryEntry = {
      ...result,
      time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    };
    setHistory((prev) => [entry, ...prev].slice(0, 25));
  };

  const labelDotClass = (l?: string) => {
    if (l === 'HUMAN') return 'score-dot-human';
    if (l === 'SYNTHETIC') return 'score-dot-synthetic';
    return 'score-dot-uncertain';
  };

  if (!themeReady) return null;

  /* ═══════════════════════════════════════════════════════════════════════════
     Render Dashboard
     ═══════════════════════════════════════════════════════════════════════════ */
  return (
    <div className="max-w-[1200px] mx-auto px-3 sm:px-5 py-4 sm:py-6 pb-14">
      <audio ref={audioElRef} onEnded={() => setIsPlaying(false)} />

      {/* ─── Top Header ─── */}
      <header className="flex items-center justify-between mb-5 flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="w-[44px] h-[44px] sm:w-[50px] sm:h-[50px] rounded-2xl flex items-center justify-center shadow-md shrink-0"
               style={{ background: 'linear-gradient(135deg, var(--forest), var(--olive))', border: '1px solid var(--card-border-subtle)' }}>
            <Shield size={24} className="text-[var(--clay)]" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-[1.35rem] sm:text-[1.65rem] font-[900] tracking-tight text-[var(--text-primary)]">
                VoiceGuard <span className="text-[var(--clay)]">AI</span>
              </h1>
              <span className="info-chip text-[0.6rem] sm:text-[0.62rem] uppercase font-extrabold tracking-wider">
                Acoustic Forensics
              </span>
            </div>
            <p className="text-[0.7rem] sm:text-[0.76rem] font-bold text-muted">
              Deepfake Voice Clone Detection & Acoustic Prosody Telemetry
            </p>
          </div>
        </div>

        {/* Right Header Badges */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="clay-btn clay-btn-sm cursor-default text-[0.68rem] sm:text-[0.72rem]" style={{ padding: '6px 12px' }}>
            {isRecording ? <Radio size={13} className="text-[var(--verdict-human)] animate-pulse" /> : <Wifi size={13} className="text-[var(--olive)]" />}
            <span className="font-bold">
              {isRecording ? 'Capturing PCM' : 'Pipeline Online'}
            </span>
          </div>

          {history.length > 0 && (
            <button onClick={exportReport} className="clay-btn clay-btn-sm" title="Download Forensic JSON Audit">
              <FileText size={13} className="text-[var(--clay)]" />
              <span className="text-[0.68rem] sm:text-[0.72rem] font-bold">Export Audit</span>
            </button>
          )}

          {/* HuggingFace Wav2Vec2 API Key Button */}
          <button
            onClick={() => setShowHfModal(true)}
            className="clay-btn clay-btn-sm"
            title="Configure HuggingFace Wav2Vec2 API Key"
          >
            <Key size={13} className={hfKey ? "text-[var(--verdict-human)]" : "text-[var(--clay)]"} />
            <span className="text-[0.68rem] sm:text-[0.72rem] font-bold">
              {hfKey ? 'HF AI Connected' : 'Set HF Key'}
            </span>
          </button>

          {/* Theme Toggle (Light <-> Dark) */}
          <button onClick={toggleDarkMode} className="theme-toggle" aria-label="Toggle theme mode" title="Switch Theme">
            <div className={`theme-toggle-icon ${darkMode ? 'spin' : ''}`}>
              {darkMode ? <Sun size={18} className="text-[var(--clay)]" /> : <Moon size={18} className="text-[var(--text-primary)]" />}
            </div>
          </button>
        </div>
      </header>

      {/* ─── Threat Alert Banner ─── */}
      {typeof trustScore === 'number' && trustScore <= 0.38 && (
        <div className="alert-banner alert-danger mb-5">
          <ShieldAlert size={22} className="text-[var(--verdict-synthetic)] shrink-0" />
          <div>
            <span className="font-extrabold">SYNTHETIC DEEPFAKE ATTACK IDENTIFIED:</span>
            <span className="ml-1.5 font-semibold">
              Robotic prosody or neural vocoder artifacts detected with {((1 - trustScore) * 100).toFixed(0)}% confidence
            </span>
          </div>
        </div>
      )}
      {typeof trustScore === 'number' && trustScore >= 0.52 && label === 'HUMAN' && (
        <div className="alert-banner alert-safe mb-5">
          <ShieldCheck size={22} className="text-[var(--verdict-human)] shrink-0" />
          <div>
            <span className="font-extrabold">AUTHENTIC HUMAN VOICE VERIFIED:</span>
            <span className="ml-1.5 font-semibold">
              Biological prosody dynamic and natural vocal tract resonance validated (Trust: {(trustScore * 100).toFixed(0)}%)
            </span>
          </div>
        </div>
      )}

      {/* ─── HERO SECTION: Trust Gauge + Unified Ingestion Studio ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-[330px_1fr] gap-5 mb-5">
        
        {/* Left Column: Trust Score & Verdict */}
        <div className="clay-card flex flex-col items-center justify-between py-6">
          <div className="card-header w-full justify-between">
            <div className="flex items-center gap-2">
              <div className="card-icon" style={{ background: 'var(--olive-bg)', color: 'var(--olive)' }}>
                <Gauge size={16} />
              </div>
              <span>Trust Index</span>
            </div>
            <span className="info-chip text-[0.65rem]">{modelName}</span>
          </div>

          <TrustGauge score={trustScore} label={label} />

          {/* Probability Split Bar */}
          <div className="w-full mt-3 px-2">
            <div className="flex justify-between text-[0.68rem] font-bold text-muted mb-1">
              <span className="text-[var(--verdict-human)] font-extrabold">
                Human: {typeof trustScore === 'number' ? `${Math.round(trustScore * 100)}%` : '—'}
              </span>
              <span className="text-[var(--verdict-synthetic)] font-extrabold">
                Synthetic: {typeof trustScore === 'number' ? `${Math.round((1 - trustScore) * 100)}%` : '—'}
              </span>
            </div>
            <div className="w-full h-2.5 rounded-full overflow-hidden flex bg-[var(--canvas-bg)] border border-[var(--card-border-subtle)]">
              <div style={{ width: `${(trustScore ?? 0.5) * 100}%`, background: 'var(--verdict-human)', transition: 'width 0.6s ease' }} />
              <div style={{ width: `${(1 - (trustScore ?? 0.5)) * 100}%`, background: 'var(--verdict-synthetic)', transition: 'width 0.6s ease' }} />
            </div>
          </div>

          {/* Latency & Window Chips */}
          <div className="flex items-center justify-center gap-2 mt-4 flex-wrap">
            <span className="info-chip"><Zap size={11} /> {inferenceMs ? `${inferenceMs} ms` : '< 20 ms'}</span>
            <span className="info-chip"><Package size={11} /> {bufferSec ? `${bufferSec.toFixed(1)}s buffer` : '16 kHz mono'}</span>
          </div>
        </div>

        {/* Right Column: Ingestion Studio (Visualizer + Tactile Controls) ─── */}
        <div className="clay-card flex flex-col justify-between min-h-[360px]">
          {/* Studio Header & Visualizer Switcher */}
          <div className="card-header justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="card-icon" style={{ background: 'var(--clay-bg)', color: 'var(--clay)' }}>
                <Activity size={16} />
              </div>
              <span>Acoustic Visualizer & Ingestion Studio</span>
            </div>

            <div className="clay-tabs">
              <button
                onClick={() => setVisualMode('waveform')}
                className={`clay-tab-item ${visualMode === 'waveform' ? 'active' : ''}`}
              >
                Oscilloscope
              </button>
              <button
                onClick={() => setVisualMode('spectrum')}
                className={`clay-tab-item ${visualMode === 'spectrum' ? 'active' : ''}`}
              >
                Mel-Spectrogram
              </button>
            </div>
          </div>

          {/* Visualizer Canvas */}
          <div className="mb-4">
            <WaveformCanvas
              analyserNode={analyserNode}
              isActive={isRecording}
              trustScore={trustScore}
              visualMode={visualMode}
            />
          </div>

          {/* Bottom Dual Ingestion Bar: Record + File Upload Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-[var(--card-border-subtle)]">
            
            {/* Action 1: Live Mic Recording */}
            <div className="flex items-center gap-3">
              <div className="relative">
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`record-btn ${isRecording ? 'recording' : ''}`}
                  style={{ width: '56px', height: '56px' }}
                  title={isRecording ? 'Stop Recording' : 'Start Recording'}
                >
                  {isRecording ? <Square size={20} fill="currentColor" /> : <Mic size={22} />}
                </button>
                <div className="record-ring" style={{ inset: '-5px' }} />
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <span className="text-[1.15rem] font-[900] tabular-nums text-[var(--text-primary)]">
                    {formatTime(recordingTime)}
                  </span>
                  {isRecording && (
                    <span className="info-chip text-[0.6rem] text-[var(--verdict-synthetic)] font-bold">
                      STREAMING
                    </span>
                  )}
                </div>
                <p className="text-[0.72rem] font-bold text-muted">
                  {isRecording ? 'Recording microphone feed...' : 'Click mic to record & analyze'}
                </p>

                {/* Post-recording actions */}
                {hasRecording && !isRecording && (
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <button
                      onClick={() => recordedBlob && analyzeBlob(recordedBlob, 'mic_recording.webm')}
                      disabled={isAnalyzingRecord}
                      className="clay-btn clay-btn-olive clay-btn-sm text-[0.7rem] py-1 px-2.5"
                    >
                      {isAnalyzingRecord ? (
                        <RefreshCw size={12} className="animate-spin" />
                      ) : (
                        <><Sparkles size={12} /> Re-Analyze</>
                      )}
                    </button>
                    <button onClick={isPlaying ? pausePlayback : playRecording} className="clay-btn clay-btn-sm text-[0.7rem] py-1 px-2.5">
                      {isPlaying ? <Pause size={12} /> : <Play size={12} />}
                    </button>
                    <button onClick={downloadRecording} className="clay-btn clay-btn-sm text-[0.7rem] py-1 px-2.5">
                      <Download size={12} />
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Action 2: File Upload Zone */}
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />
              <div
                className={`dropzone py-3 px-3 flex items-center justify-center gap-3 ${dragOver ? 'dragover' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file) handleFileSelect(file);
                }}
              >
                <FileAudio size={24} className="text-[var(--clay)] shrink-0" />
                <div className="text-left">
                  <p className="text-[0.76rem] font-bold text-[var(--text-secondary)]">
                    {isUploading ? 'Extracting Acoustic Features...' : selectedFile ? selectedFile.name : 'Drop audio file or browse'}
                  </p>
                  <p className="text-[0.66rem] text-muted font-semibold">
                    Supports .WAV, .MP3, .OGG, .WEBM, .FLAC
                  </p>
                </div>
              </div>
            </div>

          </div>
        </div>

      </div>

      {/* ─── Forensic Acoustic Diagnostics Telemetry Grid ─── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
        {/* Pitch Dynamic */}
        <div className="metric-tile">
          <span className="metric-title">F0 Pitch Dynamic</span>
          <span className="metric-val">
            {features?.pitch_std !== undefined && features.pitch_std > 0 ? `±${features.pitch_std.toFixed(1)} Hz` : '—'}
          </span>
          <span className="metric-desc">
            {features?.pitch_std !== undefined && features.pitch_std > 0
              ? features.pitch_std < 12
                ? '⚠ Monotone (Synthetic)'
                : '✓ Dynamic Prosody'
              : 'Pitch Variation'}
          </span>
        </div>

        {/* Spectral Centroid */}
        <div className="metric-tile">
          <span className="metric-title">Spectral Centroid</span>
          <span className="metric-val">
            {features?.spectral_centroid ? `${Math.round(features.spectral_centroid)} Hz` : '—'}
          </span>
          <span className="metric-desc">
            {features?.spectral_centroid !== undefined
              ? features.spectral_centroid > 4000
                ? 'Elevated Brightness'
                : 'Natural Formant Curve'
              : 'Vocal Brightness'}
          </span>
        </div>

        {/* Energy Dynamics */}
        <div className="metric-tile">
          <span className="metric-title">Dynamic Energy</span>
          <span className="metric-val">
            {features?.energy_rms ? `${(features.energy_rms * 100).toFixed(1)} RMS` : '—'}
          </span>
          <span className="metric-desc">
            {features?.energy_std !== undefined
              ? features.energy_std < 0.005
                ? '⚠ Constant Loudness'
                : '✓ Syllable Modulation'
              : 'Speech Modulation'}
          </span>
        </div>

        {/* Spectral Flatness */}
        <div className="metric-tile">
          <span className="metric-title">Spectral Flatness</span>
          <span className="metric-val">
            {features?.spectral_flatness !== undefined ? features.spectral_flatness.toFixed(3) : '—'}
          </span>
          <span className="metric-desc">
            {features?.spectral_flatness !== undefined
              ? features.spectral_flatness > 0.28
                ? 'High Noise Diffusion'
                : '✓ Harmonic Resonance'
              : 'Wiener Entropy'}
          </span>
        </div>
      </div>

      {/* ─── Forensic Acoustic Evidence Breakdown ─── */}
      <div className="clay-card mb-5">
        <div className="card-header">
          <div className="card-icon" style={{ background: 'var(--olive-bg)', color: 'var(--olive)' }}>
            <Tags size={16} />
          </div>
          Acoustic Forensic Evidence
        </div>
        <div className="flex flex-wrap gap-2">
          {reasons.length === 0 ? (
            <span className="clay-tag text-muted">
              Record microphone or upload voice sample to inspect forensic acoustic metrics
            </span>
          ) : (
            reasons.map((reason, i) => {
              const isNatural = reason.includes('[NATURAL]') || reason.includes('✓');
              const isAnomaly = reason.includes('[ANOMALY]') || reason.includes('⚠');
              const cls = isNatural ? 'clay-tag-good' : isAnomaly ? 'clay-tag-bad' : 'clay-tag-warn';
              const cleanReason = reason.replace(/\[(NATURAL|ANOMALY|SUSPICIOUS)\]|[✓⚠️]/g, '').trim();
              const Icon = isNatural ? CheckCircle : isAnomaly ? AlertTriangle : Clock;
              return (
                <span key={i} className={`clay-tag ${cls}`}>
                  <Icon size={12} />
                  {cleanReason}
                </span>
              );
            })
          )}
        </div>
      </div>

      {/* ─── Forensic Session Audit Log ─── */}
      <div className="clay-card max-h-[300px] overflow-y-auto">
        <div className="card-header justify-between">
          <div className="flex items-center gap-2">
            <div className="card-icon" style={{ background: 'var(--olive-bg)', color: 'var(--olive)' }}>
              <History size={16} />
            </div>
            <span>Session Audit History</span>
          </div>
          {history.length > 0 && (
            <button onClick={() => setHistory([])} className="clay-btn clay-btn-sm text-[0.68rem] py-1 px-3">
              Clear History
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div className="text-center py-6 text-muted font-bold text-[0.85rem]">
            <Volume2 size={30} className="mx-auto mb-2 opacity-35" />
            No audit sessions logged yet. Record from microphone or load an audio sample to start.
          </div>
        ) : (
          <div className="overflow-x-auto w-full">
            <table className="history-table min-w-[500px]">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Trust Score</th>
                  <th>Classification</th>
                  <th>Duration</th>
                  <th>Latency</th>
                  <th>Detection Model</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i}>
                    <td>{h.time}</td>
                    <td>
                      <span className={`score-dot ${labelDotClass(h.label)}`} />
                      {typeof h.score === 'number' ? `${Math.round(h.score * 100)}%` : '—'}
                    </td>
                    <td className="font-[800]">{h.label}</td>
                    <td>{h.duration ? `${h.duration}s` : '—'}</td>
                    <td>{h.inference_ms ? `${h.inference_ms}ms` : '—'}</td>
                    <td className="text-muted">{h.model || 'heuristic-v2'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ─── Professional Footer ─── */}
      <footer className="mt-8 pt-6 border-t" style={{ borderColor: 'var(--card-border-subtle)' }}>
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <div className="space-y-1">
            <div className="flex items-center justify-center sm:justify-start gap-2">
              <Shield size={14} className="text-[var(--clay)]" />
              <span className="text-[0.78rem] font-extrabold text-[var(--text-primary)] tracking-tight">
                VoiceGuard <span className="text-[var(--clay)]">AI</span>
              </span>
              <span className="text-[0.6rem] font-bold px-2 py-0.5 rounded-full" style={{
                background: 'var(--olive-bg)', color: 'var(--olive)', border: '1px solid var(--card-border-subtle)'
              }}>v2.0</span>
            </div>
            <p className="text-[0.68rem] text-muted font-semibold">
              Built by <span className="text-[var(--text-secondary)] font-extrabold">Ronan S Atomos</span>
              <span className="mx-1.5 opacity-40">•</span>
              IIT BHU Hackathon 2026
            </p>
          </div>

          <div className="flex items-center gap-3 text-[0.66rem] font-bold text-muted">
            <button
              onClick={() => setShowTerms(!showTerms)}
              className="hover:text-[var(--text-secondary)] transition-colors underline underline-offset-2 decoration-dotted"
            >
              Terms & Conditions
            </button>
            <span className="opacity-30">•</span>
            <span>Acoustic Forensics Engine</span>
          </div>
        </div>

        {/* ─── Terms & Conditions Expandable ─── */}
        {showTerms && (
          <div
            className="mt-4 p-4 rounded-2xl text-[0.72rem] leading-relaxed space-y-2 font-medium"
            style={{
              background: 'var(--canvas-bg)',
              border: '1px solid var(--card-border-subtle)',
              color: 'var(--text-muted)',
            }}
          >
            <h4 className="text-[0.78rem] font-extrabold text-[var(--text-primary)] tracking-tight mb-2">
              Terms & Conditions
            </h4>
            <p>
              <strong>1. Purpose.</strong> VoiceGuard AI is an acoustic forensics research tool designed for educational and demonstration purposes as part of IIT BHU Hackathon 2026. It is not intended as a definitive legal or forensic instrument.
            </p>
            <p>
              <strong>2. No Guarantee of Accuracy.</strong> While VoiceGuard AI employs advanced signal processing, machine learning classifiers, and optional deep neural network (Wav2Vec2) inference to distinguish human speech from synthetic audio, no detection system is 100% accurate. Results should be interpreted as probabilistic assessments, not absolute determinations.
            </p>
            <p>
              <strong>3. Data Privacy.</strong> Audio recordings submitted for analysis are processed transiently and are not stored, logged, or transmitted to any third party beyond the optional HuggingFace Inference API (when a user-provided API key is configured). No personally identifiable voice biometrics are retained.
            </p>
            <p>
              <strong>4. Limitation of Liability.</strong> The developer (Ronan S Atomos) and contributors shall not be held liable for any decisions, actions, or consequences arising from the use of this tool or its outputs.
            </p>
            <p>
              <strong>5. Open Source.</strong> This project is provided as-is for educational and research use. Unauthorized commercial redistribution is prohibited without explicit permission.
            </p>
            <button
              onClick={() => setShowTerms(false)}
              className="mt-2 text-[var(--clay)] font-bold hover:opacity-80 transition-opacity text-xs"
            >
              Close ✕
            </button>
          </div>
        )}

        <p className="text-center text-[0.62rem] text-muted mt-4 pb-2 font-semibold opacity-60">
          © {new Date().getFullYear()} Ronan S Atomos. All rights reserved.
        </p>
      </footer>

      {/* ─── HuggingFace API Key Modal ─── */}
      {showHfModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div 
            className="w-full max-w-md rounded-3xl p-6 shadow-2xl space-y-4 border"
            style={{
              background: 'var(--card-bg, #1a241c)',
              borderColor: 'var(--card-border, #2d3b2f)',
              color: 'var(--text-primary, #e6ede8)'
            }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Key className="text-[var(--clay)]" size={20} />
                <h3 className="text-lg font-extrabold tracking-tight">Hugging Face API Key</h3>
              </div>
              <button 
                onClick={() => setShowHfModal(false)}
                className="p-1.5 rounded-xl hover:bg-white/10 text-muted transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            <p className="text-xs text-muted leading-relaxed">
              Connect VoiceGuard AI directly to the <span className="text-[var(--clay)] font-semibold">Wav2Vec2 Foundation Model</span> (<code className="text-[11px] px-1 py-0.5 rounded bg-black/30">MelodyMachine/Deepfake-audio-detection-V2</code>) for deep neural network deepfake classification.
            </p>

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-muted block">Hugging Face Access Token (Read)</label>
              <input
                type="password"
                value={hfInput}
                onChange={(e) => setHfInput(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                className="w-full px-3.5 py-2.5 rounded-xl text-sm outline-none border focus:border-[var(--clay)] transition-colors"
                style={{
                  background: 'rgba(0,0,0,0.25)',
                  borderColor: 'var(--card-border, #2d3b2f)',
                  color: 'var(--text-primary, #e6ede8)'
                }}
              />
              <p className="text-[11px] text-muted">
                Get a free token at{' '}
                <a 
                  href="https://huggingface.co/settings/tokens" 
                  target="_blank" 
                  rel="noreferrer"
                  className="text-[var(--clay)] underline hover:opacity-80 font-bold"
                >
                  huggingface.co/settings/tokens
                </a>
              </p>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              {hfKey && (
                <button
                  type="button"
                  onClick={() => {
                    setHfInput('');
                    setHfKey('');
                    if (typeof window !== 'undefined') localStorage.removeItem('voiceguard_hf_key');
                    setShowHfModal(false);
                  }}
                  className="clay-btn clay-btn-sm text-xs font-bold text-[var(--verdict-synthetic)]"
                >
                  Remove Key
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowHfModal(false)}
                className="clay-btn clay-btn-sm text-xs font-bold"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={saveHfKey}
                className="clay-btn clay-btn-sm clay-btn-primary text-xs font-extrabold"
                style={{ background: 'var(--clay)', color: '#fff' }}
              >
                Save & Connect
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
