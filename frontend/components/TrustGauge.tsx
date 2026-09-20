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

  const getStrokeColor = (): string => {
    if (numScore === null) return 'var(--verdict-uncertain)';
    if (numScore >= 0.52) return 'var(--verdict-human)';
    if (numScore <= 0.38) return 'var(--verdict-synthetic)';
    return 'var(--verdict-uncertain)';
  };

  const getLabelClass = (): string => {
    if (numScore === null) return 'text-[var(--text-muted)] bg-[var(--canvas-bg)] border border-[var(--card-border-subtle)]';
    if (numScore >= 0.52) return 'text-[var(--verdict-human)] bg-[var(--verdict-human-bg)] border border-[var(--card-border)]';
    if (numScore <= 0.38) return 'text-[var(--verdict-synthetic)] bg-[var(--verdict-synthetic-bg)] border border-[var(--card-border)]';
    return 'text-[var(--verdict-uncertain)] bg-[var(--verdict-uncertain-bg)] border border-[var(--card-border-subtle)]';
  };

  const getVerdictSubtitle = (): string => {
    if (numScore === null) return 'Awaiting Audio Ingestion';
    if (numScore >= 0.52) return 'Human Biological Prosody';
    if (numScore <= 0.38) return 'Neural Synthetic Artifacts';
    return 'Borderline Acoustic Dynamics';
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-[195px] h-[195px] mb-2">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 200 200">
          <circle className="gauge-bg" cx="100" cy="100" r={radius} />
          <circle
            className="gauge-fill"
            cx="100"
            cy="100"
            r={radius}
            stroke={getStrokeColor()}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="gauge-score">
            {numScore !== null ? `${Math.round(numScore * 100)}%` : '—'}
          </span>
          <span className={`gauge-label ${getLabelClass()}`}>{label || 'WAITING'}</span>
        </div>
      </div>
      <p className="text-[0.72rem] font-bold text-center text-muted tracking-wide">
        {getVerdictSubtitle()}
      </p>
    </div>
  );
}
