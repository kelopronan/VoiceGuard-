'use client';

import { useEffect, useRef } from 'react';

interface WaveformCanvasProps {
  analyserNode: AnalyserNode | null;
  isActive: boolean;
  trustScore: number | null | undefined;
  visualMode?: 'waveform' | 'spectrum';
}

export default function WaveformCanvas({
  analyserNode,
  isActive,
  trustScore,
  visualMode = 'waveform',
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) {
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }
    };
    resize();
    window.addEventListener('resize', resize);

    const getPrimaryColor = (): string => {
      const isDark = document.documentElement.classList.contains('dark');
      if (isDark) {
        // Dark Mode: Olive Leaf, Sunlit Clay, Copperwood
        if (trustScore === null || trustScore === undefined) return '#DDA15E';
        if (trustScore >= 0.52) return '#738344';
        if (trustScore <= 0.38) return '#BC6C25';
        return '#DDA15E';
      } else {
        // Light Mode: Pacific Blue, Blue Slate, Tangerine Dream
        if (trustScore === null || trustScore === undefined) return '#5C9EAD';
        if (trustScore >= 0.52) return '#5C9EAD';
        if (trustScore <= 0.38) return '#E39774';
        return '#326273';
      }
    };

    const draw = () => {
      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;
      ctx.clearRect(0, 0, w, h);

      const isDark = document.documentElement.classList.contains('dark');

      // Grid lines
      ctx.strokeStyle = isDark ? 'rgba(96, 108, 56, 0.1)' : 'rgba(92, 158, 173, 0.12)';
      ctx.lineWidth = 1;
      const gridSteps = 6;
      for (let i = 1; i < gridSteps; i++) {
        const y = (h / gridSteps) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      if (!analyserNode || !isActive) {
        // Idle baseline
        ctx.strokeStyle = isDark ? 'rgba(221, 161, 94, 0.25)' : 'rgba(92, 158, 173, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      const color = getPrimaryColor();

      if (visualMode === 'waveform') {
        const bufferLength = analyserNode.fftSize;
        const dataArray = new Uint8Array(bufferLength);
        analyserNode.getByteTimeDomainData(dataArray);

        // Glow
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = color;
        ctx.beginPath();

        const sliceWidth = w / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * h) / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }

        ctx.lineTo(w, h / 2);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Bottom spectrum bars
        const freqData = new Uint8Array(analyserNode.frequencyBinCount);
        analyserNode.getByteFrequencyData(freqData);

        const barCount = 48;
        const barWidth = w / barCount;
        const step = Math.floor(freqData.length / barCount);

        for (let i = 0; i < barCount; i++) {
          const value = freqData[i * step];
          const barHeight = (value / 255) * (h * 0.35);

          ctx.fillStyle = `${color}35`;
          ctx.beginPath();
          ctx.roundRect(
            i * barWidth + 1.5,
            h - barHeight,
            barWidth - 3,
            barHeight,
            [3, 3, 0, 0]
          );
          ctx.fill();
        }
      } else {
        // Mel-Spectrogram Heatmap
        const freqData = new Uint8Array(analyserNode.frequencyBinCount);
        analyserNode.getByteFrequencyData(freqData);

        const barCount = 64;
        const barWidth = w / barCount;
        const step = Math.floor(freqData.length / barCount);

        for (let i = 0; i < barCount; i++) {
          const value = freqData[i * step];
          const barHeight = (value / 255) * (h * 0.85);

          const grad = ctx.createLinearGradient(0, h, 0, h - barHeight);
          if (isDark) {
            grad.addColorStop(0, '#606C38');
            grad.addColorStop(0.6, '#DDA15E');
            grad.addColorStop(1, '#BC6C25');
          } else {
            grad.addColorStop(0, '#5C9EAD');
            grad.addColorStop(0.5, '#326273');
            grad.addColorStop(1, '#E39774');
          }

          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.roundRect(
            i * barWidth + 1.5,
            h - barHeight,
            barWidth - 3,
            barHeight,
            [4, 4, 0, 0]
          );
          ctx.fill();
        }
      }

      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [analyserNode, isActive, trustScore, visualMode]);

  return (
    <div className="waveform-wrap">
      <canvas ref={canvasRef} />
      <div className="waveform-badge">
        {isActive ? '● Live Ingestion' : 'Standby'}
      </div>
    </div>
  );
}
