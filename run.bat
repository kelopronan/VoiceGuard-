@echo off
title VoiceGuard AI - Launcher
echo ===================================================
echo     VoiceGuard AI - Real-Time Voice Clone Detector
echo ===================================================
echo.

echo [*] Starting Backend server on http://localhost:8000 ...
start "VoiceGuard Backend (FastAPI)" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 >nul

echo [*] Starting Frontend Next.js app on http://localhost:3000 ...
start "VoiceGuard Frontend (Next.js)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo [!] Both servers are starting up!
echo     - Backend API & WebSocket: http://localhost:8000
echo     - Frontend Web App:        http://localhost:3000
echo.
echo You can open your browser to http://localhost:3000
echo ===================================================
pause
