'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Shield, ShieldCheck, ShieldAlert, Mic, Square, Upload, Play, Pause, Download,
  Activity, BarChart3, Tags, Zap, Bot, Package,
  AlertTriangle, CheckCircle, XCircle, Sun, Moon,
  FileAudio, History, Clock, Wifi, WifiOff, Volume2, X, Sparkles,
  Radio, FileText, Gauge, RefreshCw, AudioWaveform
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
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || '';
const API_BASE = BACKEND_URL ? `${BACKEND_URL}/api` : '/api';
const WS_URL = BACKEND_URL
  ? `${BACKEND_URL.replace(/^http/, 'ws')}/ws/stream`
  : 'ws://localhost:8000/ws/stream';

/* ─── Audio Helpers ───────────────────────────────────────────────────────── */
function float32ToBase64(float32Array: Float32Array): string {
  const bytes = new Uint8Array(float32Array.buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

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

  /* ─── Refs ─── */
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaRecRef = useRef<MediaRecorder | null>(null);
  const recChunksRef = useRef<Blob[]>([]);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const allPcmChunksRef = useRef<Float32Array[]>([]);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
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
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData });
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

      pcmChunksRef.current = [];
      allPcmChunksRef.current = [];

      processor.onaudioprocess = (e) => {
        const data = e.inputBuffer.getChannelData(0);
        const copy = new Float32Array(data);
        pcmChunksRef.current.push(copy);
        allPcmChunksRef.current.push(copy);
      };

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';
      const mediaRec = new MediaRecorder(stream, { mimeType });
      recChunksRef.current = [];
      mediaRec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recChunksRef.current.push(e.data);
      };
      mediaRec.start(250);
      mediaRecRef.current = mediaRec;

      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          try {
            const result: AnalysisResult = JSON.parse(event.data);
            if (typeof result.score === 'number') {
              setTrustScore(result.score);
              setLabel(result.label);
              setReasons(result.reasons || []);
              setFeatures(result.features || {});
              setInferenceMs(result.inference_ms || 0);
              setBufferSec(result.buffer_seconds || 0);
              setModelName(result.model || 'heuristic-v2');
            }
          } catch { /* ignore */ }
        };

        ws.onopen = () => {
          sendIntervalRef.current = setInterval(() => {
            if (pcmChunksRef.current.length === 0 || ws.readyState !== WebSocket.OPEN) return;
            const totalLen = pcmChunksRef.current.reduce((s, a) => s + a.length, 0);
            const combined = new Float32Array(totalLen);
            let off = 0;
            for (const c of pcmChunksRef.current) { combined.set(c, off); off += c.length; }
            pcmChunksRef.current = [];
            const resampled = downsample(combined, audioCtx.sampleRate, 16000);
            const b64 = float32ToBase64(resampled);
            ws.send(JSON.stringify({ type: 'audio_chunk', data: b64 }));
          }, 500);
        };
      } catch (wsErr) {
        console.warn('WebSocket live stream:', wsErr);
      }

      setRecordingTime(0);
      timerRef.current = setInterval(() => setRecordingTime((t) => t + 1), 1000);

      setIsRecording(true);
      setHasRecording(false);
      setRecordedBlob(null);
      setTrustScore(null);
      setLabel('LISTENING');
      setReasons([]);
    } catch (err) {
      console.error('Mic error:', err);
      alert('Microphone access blocked. Please enable microphone permissions in your browser.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (sendIntervalRef.current) { clearInterval(sendIntervalRef.current); sendIntervalRef.current = null; }
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }

    if (wsRef.current) {
      try { wsRef.current.send(JSON.stringify({ type: 'stop' })); } catch { /* */ }
      wsRef.current.close();
      wsRef.current = null;
    }

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
      // Automatic forensic analysis on complete lossless WAV
      analyzeBlob(wavBlob, 'mic_capture.wav');
    }

    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }
    if (mediaRecRef.current && mediaRecRef.current.state !== 'inactive') {
      try { mediaRecRef.current.stop(); } catch { /* */ }
    }

    setAnalyserNode(null);
    pcmChunksRef.current = [];
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
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData });
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

      {/* ─── Footer ─── */}
      <footer className="text-center py-6 text-muted text-[0.72rem] font-bold">
        VoiceGuard AI &middot; IIT BHU Hackathon 2026 &middot; Real-Time AI Voice Cloning Detection System
      </footer>
    </div>
  );
}
