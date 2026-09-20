import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'VoiceGuard AI — Deepfake Voice Clone Detection',
  description:
    'AI-powered acoustic forensics engine that detects synthetic voice cloning, AI-generated speech, and deepfake audio attacks in real time. Built by Ronan S Atomos.',
  keywords: [
    'deepfake detection',
    'voice clone detection',
    'AI voice detection',
    'acoustic forensics',
    'VoiceGuard AI',
    'synthetic speech detection',
    'IIT BHU Hackathon',
  ],
  authors: [{ name: 'Ronan S Atomos' }],
  creator: 'Ronan S Atomos',
  openGraph: {
    title: 'VoiceGuard AI — Deepfake Voice Clone Detection',
    description:
      'Real-time AI-powered acoustic forensics engine. Detects synthetic clones, neural TTS, and deepfake audio with Wav2Vec2 + biometric analysis.',
    siteName: 'VoiceGuard AI',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'VoiceGuard AI',
    description: 'Deepfake Voice Clone Detection & Acoustic Forensics',
  },
  robots: { index: true, follow: true },
  icons: {
    icon: '/favicon.ico',
    apple: '/icon.svg',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#EEEEEE' },
    { media: '(prefers-color-scheme: dark)', color: '#151D0D' },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans antialiased transition-colors duration-500">{children}</body>
    </html>
  );
}
