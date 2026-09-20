'use client';

interface TrustGaugeProps {
  score: number | null | undefined;
  label: string;
}

export default function TrustGauge({ score, label }: TrustGaugeProps) {
  const radius = 82;
  const circumference = 2 * Math.PI * radius;
  
  const numScore = typeof score === 'number' && !isNaN(score) ? Math.max(0, Math.min(1, score)) : null;
  const offset = numScore !== null ? circumference * (1 - numScore) : circumference;

  const isRecording = label === 'RECORDING';
  const isAnalyzing = label === 'ANALYZING';

  const getStrokeColor = (): string => {
    if (isRecording) return '#0284c7';
    if (isAnalyzing) return '#f59e0b';
    if (numScore === null) return 'var(--verdict-uncertain)';
    if (numScore >= 0.52) return 'var(--verdict-human)';
    if (numScore <= 0.38) return 'var(--verdict-synthetic)';
    return 'var(--verdict-uncertain)';
  };

  const getLabelClass = (): string => {
    if (isRecording) return 'text-sky-500 bg-sky-500/10 border border-sky-500/30 animate-pulse';
    if (isAnalyzing) return 'text-amber-500 bg-amber-500/10 border border-amber-500/30 animate-pulse';
    if (numScore === null) return 'text-[var(--text-muted)] bg-[var(--canvas-bg)] border border-[var(--card-border-subtle)]';
    if (numScore >= 0.52) return 'text-[var(--verdict-human)] bg-[var(--verdict-human-bg)] border border-[var(--card-border)]';
    if (numScore <= 0.38) return 'text-[var(--verdict-synthetic)] bg-[var(--verdict-synthetic-bg)] border border-[var(--card-border)]';
    return 'text-[var(--verdict-uncertain)] bg-[var(--verdict-uncertain-bg)] border border-[var(--card-border-subtle)]';
  };

  const getVerdictSubtitle = (): string => {
    if (isRecording) return 'Listening to voice... Click Stop when finished';
    if (isAnalyzing) return 'Running 6-Pillar Forensic Battery...';
    if (numScore === null) return 'Ready to Record or Upload Audio';
    if (numScore >= 0.52) return 'Verified Biological Human Voice';
    if (numScore <= 0.38) return 'Synthetic AI Voice Clone Detected';
    return 'Borderline Acoustic Dynamics';
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-[195px] h-[195px] mb-2">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 200 200">
          <circle className="gauge-bg" cx="100" cy="100" r={radius} />
          <circle
            className={`gauge-fill ${isRecording || isAnalyzing ? 'transition-all duration-300' : ''}`}
            cx="100"
            cy="100"
            r={radius}
            stroke={getStrokeColor()}
            strokeDasharray={circumference}
            strokeDashoffset={isRecording || isAnalyzing ? circumference * 0.25 : offset}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="gauge-score">
            {isRecording ? (
              <span className="flex items-center gap-1.5 text-sky-500 text-3xl font-extrabold tracking-wider animate-pulse">
                <span className="w-3 h-3 rounded-full bg-red-500 animate-ping inline-block" />
                REC
              </span>
            ) : isAnalyzing ? (
              <span className="text-amber-500 text-3xl font-extrabold animate-pulse">...</span>
            ) : numScore !== null ? (
              `${Math.round(numScore * 100)}%`
            ) : (
              '—'
            )}
          </span>
          <span className={`gauge-label ${getLabelClass()}`}>{label || 'READY'}</span>
        </div>
      </div>
      <p className="text-[0.72rem] font-bold text-center text-muted tracking-wide">
        {getVerdictSubtitle()}
      </p>
    </div>
  );
}
