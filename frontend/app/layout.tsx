import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'VoiceGuard AI — Real-Time Voice Clone Detector',
  description: 'AI-Powered Voice Cloning Detection System for IIT BHU Hackathon',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-nunito transition-colors duration-500">{children}</body>
    </html>
  );
}
