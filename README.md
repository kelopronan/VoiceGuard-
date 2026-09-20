# VoiceGuard AI — Real-Time Voice Cloning Detection

> AI-powered system that detects deepfake/cloned voices in real-time using acoustic feature analysis and optional HuggingFace Inference API.

## Features

- **Live Microphone Recording** — Stream audio in real-time, Vocaroo-style
- **File Upload** — Drag-and-drop or browse for `.wav`, `.mp3`, `.ogg`, `.flac`, `.webm` files
- **Trust Score Gauge** — Animated circular SVG gauge (0.0 = Synthetic, 1.0 = Human)
- **Live Waveform** — Real-time audio visualization with frequency bars
- **Explainability Tags** — See _why_ the system flagged a voice (pitch, MFCC, spectral analysis)
- **Alert System** — Red/green banners for immediate verdict feedback
- **Dark Mode** — Smooth claymorphic theme toggle with persistence
- **Analysis History** — Scrollable log of all past analyses
- **HuggingFace API** — Optional API key for enhanced model inference

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, WebSockets |
| Audio Processing | librosa, numpy, soundfile |
| AI Model | Heuristic acoustic analysis + HuggingFace Inference API |
| Frontend | Next.js 14, React 18, TypeScript |
| Icons | Lucide React |
| Styling | Tailwind CSS + Custom Claymorphic CSS |
| Audio Capture | Web Audio API (ScriptProcessorNode + AnalyserNode) |

## Quick Start

### One-Click (Windows)

```bash
run.bat
```

### Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend (separate terminal):**
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

### Optional: HuggingFace API

For enhanced deepfake detection, enter a HuggingFace API key in the UI header. Get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Architecture

```
Browser Mic ──WebSocket──> FastAPI ──> Feature Extraction ──> AI Model ──> JSON Result
     │                        │            (librosa)          (heuristic     │
     │                        │                               or HF API)    │
     │<──────WebSocket────────┘<────────────────────────────────────────────┘
     │
     └──> Next.js Dashboard (Trust Gauge, Waveform, Tags, Alerts, History)
```

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server + WebSocket streaming
│   ├── audio_processor.py   # Feature extraction (MFCC, Mel-spec, pitch, energy)
│   ├── model_handler.py     # Deepfake detection (heuristic + HF API)
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main dashboard (recording, upload, dark mode)
│   │   ├── layout.tsx       # Root layout
│   │   └── globals.css      # Claymorphic design system + dark mode
│   ├── components/
│   │   ├── TrustGauge.tsx   # Animated SVG circular gauge
│   │   └── WaveformCanvas.tsx # Live audio waveform + frequency bars
│   └── package.json
├── run.bat                  # One-click Windows launcher
└── README.md
```

## Team

**IIT BHU Hackathon 2026**

---

Built with FastAPI + Next.js + Web Audio API
