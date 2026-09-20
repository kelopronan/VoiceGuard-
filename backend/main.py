"""
VoiceGuard AI — FastAPI Backend Server

Endpoints:
    GET  /api/health       → Health check + model status
    POST /api/analyze      → Upload audio file for analysis
    WS   /ws/stream        → Real-time audio streaming with sliding window
"""

import json
import time
import io
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import Optional

from model_handler import VoiceDetector
from audio_processor import AudioProcessor

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="VoiceGuard AI",
    description="Real-Time AI Voice Cloning Detection System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[*] VoiceGuard AI - Initializing...")
detector = VoiceDetector(hf_api_key=None)
processor = AudioProcessor(sr=16000)
print("[+] VoiceGuard AI server ready!")

# Sliding window audio buffers per WebSocket client
audio_buffers: dict[int, np.ndarray] = {}


# ─── Helper: Robust Audio Loading ────────────────────────────────────────────

def load_audio_from_bytes(file_bytes: bytes, filename: str = "audio.wav"):
    """
    Robust audio decoder that handles WAV, MP3, OGG, WEBM, FLAC.
    Tries soundfile, then librosa via temp file.
    """
    audio = None
    sr = 16000

    # Strategy 1: soundfile directly on bytes
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(file_bytes))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception:
                pass
        return audio.astype(np.float32), sr
    except Exception:
        pass

    # Strategy 2: imageio_ffmpeg subprocess (supports WebM, Opus, MP3, M4A, OGG, FLAC, AAC)
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-i", "pipe:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = proc.communicate(input=file_bytes)
        if len(out) > 0:
            audio = np.frombuffer(out, dtype=np.float32).copy()
            if len(audio) > 0:
                return audio, 16000
    except Exception:
        pass

    # Strategy 3: librosa with temp file
    suffix = Path(filename).suffix or ".wav"
    try:
        import librosa
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
            return audio.astype(np.float32), sr
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    except Exception:
        pass

    # Strategy 3: raw float32 or int16 PCM
    try:
        if len(file_bytes) % 4 == 0:
            audio = np.frombuffer(file_bytes, dtype=np.float32).copy()
            if len(audio) > 0 and np.max(np.abs(audio)) <= 1.5:
                return audio, 16000
        if len(file_bytes) % 2 == 0:
            audio = np.frombuffer(file_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, 16000
    except Exception:
        pass

    return None, 16000


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model": detector.model_name,
        "model_loaded": detector.model_loaded,
        "version": "1.0.0",
    }


@app.post("/api/analyze")
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    x_hf_api_key: Optional[str] = Header(None, alias="X-HF-API-Key"),
):
    """Analyze an uploaded audio file (.wav, .mp3, .ogg, .flac, .webm)."""
    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio file received."},
            )

        audio, sr = load_audio_from_bytes(contents, file.filename or "audio.wav")

        if audio is None or len(audio) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not decode audio format. Please provide a standard .wav, .mp3, or .webm file."},
            )

        start_time = time.time()
        result = detector.predict(audio, sr, hf_key=x_hf_api_key)
        inference_time = time.time() - start_time

        result["filename"] = file.filename or "recording.webm"
        result["duration"] = round(len(audio) / sr, 2)
        result["inference_ms"] = round(inference_time * 1000, 1)
        result["timestamp"] = time.time()

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"},
        )


# ─── WebSocket Streaming Endpoint ────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_audio_stream(websocket: WebSocket):
    """
    Real-time audio streaming with sliding window analysis.
    """
    await websocket.accept()
    client_id = id(websocket)
    audio_buffers[client_id] = np.array([], dtype=np.float32)
    analysis_count = 0

    print(f"[*] Client connected: {client_id}")

    try:
        while True:
            raw_data = await websocket.receive_text()

            try:
                msg = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "audio_chunk":
                audio_data = msg.get("data", "")
                if not audio_data:
                    continue

                chunk = processor.decode_audio_chunk(audio_data)
                if chunk is None or len(chunk) == 0:
                    continue

                # Sliding window buffer — keep last 3 seconds
                audio_buffers[client_id] = np.concatenate([
                    audio_buffers[client_id], chunk
                ])
                max_samples = 16000 * 3
                if len(audio_buffers[client_id]) > max_samples:
                    audio_buffers[client_id] = audio_buffers[client_id][-max_samples:]

                # Analyze when buffer has >= 500ms
                buffer_length = len(audio_buffers[client_id])
                if buffer_length >= 8000:
                    start_time = time.time()
                    result = detector.predict(audio_buffers[client_id], 16000)
                    inference_time = time.time() - start_time

                    analysis_count += 1
                    result["inference_ms"] = round(inference_time * 1000, 1)
                    result["buffer_seconds"] = round(buffer_length / 16000, 2)
                    result["analysis_count"] = analysis_count
                    result["timestamp"] = time.time()

                    await websocket.send_json(result)

            elif msg_type == "stop":
                audio_buffers[client_id] = np.array([], dtype=np.float32)
                analysis_count = 0
                await websocket.send_json({"type": "stopped"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"[*] Client disconnected: {client_id}")
    except Exception as e:
        print(f"[!] WebSocket error ({client_id}): {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        audio_buffers.pop(client_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
